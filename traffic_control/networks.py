"""Small NumPy networks with explicit gradients, suitable for CPU notebooks."""
import copy
import numpy as np
from .config import MOVEMENTS


def softmax(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -30., 30.)))


class Adam:
    def __init__(self, parameters, rate):
        self.rate = rate
        self.m = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.v = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.steps = 0

    def update(self, parameters, gradients):
        self.steps += 1
        norm = np.sqrt(sum(np.sum(g * g) for g in gradients.values()))
        scale = min(1., 10. / max(norm, 1e-12))
        for key, parameter in parameters.items():
            grad = gradients[key] * scale
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError(f'Nonfinite gradient in {key}')
            self.m[key] = .9 * self.m[key] + .1 * grad
            self.v[key] = .999 * self.v[key] + .001 * grad * grad
            first = self.m[key] / (1. - .9 ** self.steps)
            second = self.v[key] / (1. - .999 ** self.steps)
            parameter -= self.rate * first / (np.sqrt(second) + 1e-8)


class QNetwork:
    def __init__(self, seed=0, hidden=32, rate=.001):
        rng = np.random.default_rng(seed)
        self.parameters = {}
        for layer, (inputs, outputs) in enumerate(((32, hidden), (hidden, hidden), (hidden, 2))):
            self.parameters[f'w{layer}'] = rng.normal(0., np.sqrt(2. / inputs), (inputs, outputs))
            self.parameters[f'b{layer}'] = np.zeros(outputs)
        self.optimizer = Adam(self.parameters, rate)

    def forward(self, state, cache=False):
        x = np.atleast_2d(state)
        z0 = x @ self.parameters['w0'] + self.parameters['b0']
        h0 = np.maximum(z0, .01 * z0)
        z1 = h0 @ self.parameters['w1'] + self.parameters['b1']
        h1 = np.maximum(z1, .01 * z1)
        result = h1 @ self.parameters['w2'] + self.parameters['b2']
        return (result, (x, z0, h0, z1, h1)) if cache else result

    def gradients(self, state, actions, targets):
        scores, (x, z0, h0, z1, h1) = self.forward(state, cache=True)
        row = np.arange(len(x))
        error = scores[row, actions] - targets
        absolute = np.abs(error)
        loss = np.mean(np.where(absolute <= 1., .5 * error ** 2, absolute - .5))
        upstream = np.zeros_like(scores)
        upstream[row, actions] = np.clip(error, -1., 1.) / len(x)
        gradients = {'w2': h1.T @ upstream, 'b2': upstream.sum(axis=0)}
        dh1 = (upstream @ self.parameters['w2'].T) * np.where(z1 > 0, 1., .01)
        gradients.update(w1=h0.T @ dh1, b1=dh1.sum(axis=0))
        dh0 = (dh1 @ self.parameters['w1'].T) * np.where(z0 > 0, 1., .01)
        gradients.update(w0=x.T @ dh0, b0=dh0.sum(axis=0))
        return float(loss), gradients

    def train(self, state, actions, targets):
        loss, gradients = self.gradients(state, actions, targets)
        self.optimizer.update(self.parameters, gradients)
        return loss

    def clone(self):
        return copy.deepcopy(self)

    def sync(self, other):
        for key in self.parameters:
            self.parameters[key][:] = other.parameters[key]


class RegulatablePolicy:
    """Monotonic precedence scores with learnable weights and powers.

    G_a = c_a(flag) * sum_{served lane,feature} w * x**p.
    We allow signed weights. At a fixed checkpoint each partial derivative
    has a fixed sign. p is bounded between 1 and 2, including at x = 0.
    This is a restricted polynomial family inspired by Equation 1. The
    paper puts weights inside powers. Here they multiply outside powers.
    Clearance exponents are redundant for one-hot flags and are fixed at 1.
    """
    def __init__(self, rate=.003):
        self.parameters = {'weight': np.ones((2, 2, 6)),
                           'power_raw': np.full((2, 2, 6), -4.),
                           'clear_raw': np.full((2, 4), 2.)}
        self.parameters['clear_raw'][:, 0] = -.8
        self.optimizer = Adam(self.parameters, rate)

    def forward(self, state, cache=False):
        state = np.atleast_2d(state)
        features = state[:, :24].reshape(-1, 4, 6)
        x = np.stack([features[:, list(lanes), :] for lanes in MOVEMENTS], axis=1)
        flags = state[:, 24:].reshape(-1, 2, 4)
        power_sigmoid = sigmoid(self.parameters['power_raw'])
        power = 1. + power_sigmoid
        raised = x ** power[None]
        terms = raised * self.parameters['weight'][None]
        base = terms.sum(axis=(2, 3))
        clear_sigmoid = sigmoid(self.parameters['clear_raw'])
        factors = .1 + .9 * clear_sigmoid
        clearance = np.sum(flags * factors[None], axis=2)
        scores = base * clearance
        details = (x, flags, power_sigmoid, raised, base, clear_sigmoid, clearance)
        return (scores, details) if cache else scores

    def loss_and_gradients(self, states, teacher_scores, variant, actions=None, targets=None):
        scores, (x, flags, ps, raised, base, cs, clearance) = self.forward(states, cache=True)
        batch = len(states)
        if variant == 'DRQ':
            # Algorithm 1 lines 19-21 use the TD target for the sampled action.
            error = scores[np.arange(batch), actions] - targets
            loss = np.mean(error ** 2)
            upstream = np.zeros_like(scores)
            upstream[np.arange(batch), actions] = 2. * error / batch
        else:
            probabilities = softmax(scores)
            if variant == 'DRSQ':
                target = softmax(teacher_scores)
            elif variant == 'DRHQ':
                target = np.eye(2)[np.argmax(teacher_scores, axis=1)]
            else:
                raise ValueError('variant must be DRQ, DRSQ or DRHQ')
            loss = -np.mean(np.sum(target * np.log(np.maximum(probabilities, 1e-15)), axis=1))
            upstream = (probabilities - target) / batch
        multiplier = upstream * clearance
        weight_grad = np.sum(multiplier[:, :, None, None] * raised, axis=0)
        # x**p log(x) tends to zero at x=0. Avoid evaluating log(0).
        log_x = np.log(np.maximum(x, 1e-15))
        power_grad = np.sum(multiplier[:, :, None, None] * raised * log_x
                            * self.parameters['weight'][None], axis=0) * ps * (1. - ps)
        clear_grad = np.sum(upstream[:, :, None] * base[:, :, None] * flags, axis=0)
        clear_grad *= .9 * cs * (1. - cs)
        return float(loss), {'weight': weight_grad, 'power_raw': power_grad, 'clear_raw': clear_grad}

    def train(self, states, teacher_scores, variant, actions=None, targets=None):
        loss, gradients = self.loss_and_gradients(states, teacher_scores, variant, actions, targets)
        self.optimizer.update(self.parameters, gradients)
        return loss

    def explain(self, state):
        scores, (x, flags, ps, raised, base, cs, clearance) = self.forward(state, cache=True)
        contributions = (raised * self.parameters['weight'][None] * clearance[:, :, None, None])[0]
        return {'scores': scores[0].tolist(), 'action': int(np.argmax(scores[0])),
                'clearance_factors': clearance[0].tolist(),
                'contributions': contributions.tolist(),
                'powers': (1. + ps).tolist(), 'weights': self.parameters['weight'].tolist()}

    def input_derivatives(self, state):
        scores, (x, flags, ps, raised, base, cs, clearance) = self.forward(state, cache=True)
        p = 1. + ps
        return self.parameters['weight'][None] * p[None] * x ** (p[None] - 1.) * clearance[:, :, None, None]
