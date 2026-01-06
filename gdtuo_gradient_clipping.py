'''
This is modified source code from the Gradient Descent: The Ultimate Optimizer paper
that includes the proposed gradient clipping implementation.

Code wrapped in ### are the additions made to the source code for the implementation.

Code in form of
# <original line>
<updated line>
are modifications.
'''

import torch

class Optimizable:
    '''
    This is the interface for anything that has parameters that need to be
    optimized, somewhat like torch.nn.Model but with the right plumbing for
    hyperoptimizability. (Specifically, torch.nn.Model uses the Parameter
    interface which does not give us enough control about the detachments.)
    Nominal operation of an Optimizable at the lowest level is as follows:
        o = MyOptimizable(...)
        o.initialize()
        loop {
            o.begin()
            o.zero_grad()
            loss = --compute loss function from parameters--
            loss.backward()
            o.step()
        }
    Optimizables recursively handle updates to their optimiz*ers*.
    '''
    # def __init__(self, parameters, optimizer)
    def __init__(self, parameters, optimizer, parameter_g_norms=None):
        self.parameters = parameters # a dict mapping names to tensors
        self.optimizer = optimizer   # which must itself be Optimizable
        ###
        self.parameter_g_norms = parameter_g_norms
        ###
        self.all_params_with_gradients = []

    def initialize(self):
        ''' Initialize parameters, e.g. with a Kaiming initializer. '''
        pass

    def begin(self):
        ''' Enable gradient tracking on current parameters. '''
        for param in self.all_params_with_gradients:
             param.grad = None
        self.all_params_with_gradients.clear()
        for name, param in self.parameters.items():
            param.requires_grad_() # keep gradient information...
            param.retain_grad()    # even if not a leaf...
            self.all_params_with_gradients.append(param)
        self.optimizer.begin()

    def zero_grad(self):
        ''' Set all gradients to zero. '''
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()

    ''' Note: at this point you would probably call .backwards() on the loss
    function. '''

    def step(self):
        ''' Update parameters '''
        pass
    ###
    def print_parameter_gradient_info(self):
        if isinstance(self.optimizer, NoOpOptimizer):
            print(str(self) + "has no optimizer of its parameters.")
        else:
            print(str(self) + "parameter gradient info:")
            for name in self.parameter_g_norms:
                print("parameter {}:".format(name))
                g_norms, average_g_norm = self.parameter_g_norms[name]
                print("\tmaximum gradient norm: {}".format(max(g_norms)))
                print("\taverage gradient norm: {}".format(average_g_norm))
    ###

class NoOpOptimizer(Optimizable):
    '''
    NoOpOptimizer sits on top of a stack, and does not affect what lies below.
    '''
    def __init__(self):
        pass

    def initialize(self):
        pass

    def begin(self):
        pass

    def zero_grad(self):
        pass

    # def step(self, params):
    def step(self, params, param_g_norms=None):
        pass

    def __str__(self):
        return ''

class SGD(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''
    # def __init__(self, alpha=0.01, mu=0.0, optimizer=NoOpOptimizer()):
    def __init__(self, alpha=0.01, mu=0.0, clip=False, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu)
        }
        ###
        self.clip = clip
        parameter_g_norms = None
        if not isinstance(optimizer, NoOpOptimizer):
            parameter_g_norms = {
                'alpha': [[], 0],
                'mu': [[], 0],
            }
        ###
        # super().__init__(parameters, optimizer)
        super().__init__(parameters, optimizer, parameter_g_norms)

    # def step(self, params):
    def step(self, params, param_g_norms=None):
        # self.optimizer.step(self.parameters)
        self.optimizer.step(self.parameters, self.parameter_g_norms)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            if self.mu != 0.0:
                if name not in self.state:
                    buf = self.state[name] = g
                else:
                    buf = self.state[name].detach()
                    buf = buf * self.parameters['mu'] + g
                g = self.state[name] = buf
            ###
            if param_g_norms:
                # only clip if SGD is an optimizer of another optimizer's parameters
                g = update_param_g_norm_info(param_g_norms, name, g, self.clip)
            ###
            params[name] = p - g * self.parameters['alpha']

    def __str__(self):
        return 'sgd / '+ str(self.optimizer)

class SGDPerParam(Optimizable):
    '''
    Optimizes parameters individually with SGD.
    '''
    def __init__(self, params, optimizer=NoOpOptimizer()):
        parameters = {k + '_alpha' : torch.tensor(v) for k, v in params}
        ###
        parameter_g_norms = None
        if not isinstance(optimizer, NoOpOptimizer):
            parameter_g_norms = {k + '_alpha' : [[], 0] for k, v in params}
        ###
        # super().__init__(parameters, optimizer)
        super().__init__(parameters, optimizer, parameter_g_norms)

    # def step(self, params):
    def step(self, params, param_g_norms=None):
        # self.optimizer.step(self.parameters)
        self.optimizer.step(self.parameters, self.parameter_g_norms)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            ###
            if param_g_norms:
                g = update_param_g_norm_info(param_g_norms, name, g)
            ###
            if name + '_alpha' not in self.parameters: params[name] = p
            else: params[name] = p - g * self.parameters[name + '_alpha']

    def __str__(self):
        return 'sgdPerParam / ' + str(self.optimizer)

class AdaGrad(Optimizable):
    '''
    A hyperoptimizable AdaGrad.
    '''
    def __init__(self, alpha=0.01, optimizer=NoOpOptimizer()):
        self.eps = 1e-10
        self.cache = {}
        parameters = {
            'alpha': torch.tensor(alpha)
        }
        ###
        parameter_g_norms = None
        if not isinstance(optimizer, NoOpOptimizer):
            parameter_g_norms = {
                'alpha': [[], 0]
            }
        ###
        # super().__init__(parameters, optimizer)
        super().__init__(parameters, optimizer, parameter_g_norms)

    # def step(self, params):
    def step(self, params, param_g_norms=None):
        # self.optimizer.step(self.parameters)
        self.optimizer.step(self.parameters, self.parameter_g_norms)
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'G': torch.zeros_like(param) + 1e-1
                }
            g = param.grad.detach()
            ###
            if param_g_norms:
                g = update_param_g_norm_info(param_g_norms, name, g)
            ###
            self.cache[name]['G'] = G = self.cache[name]['G'].detach() + torch.square(g)
            params[name] = param.detach() - self.parameters['alpha'] * g / torch.sqrt(G + self.eps).detach()

    def __str__(self):
        return 'adagrad / ' + str(self.optimizer)

class RMSProp(Optimizable):
    '''
    A hyperoptimizable RMSProp.
    '''
    def clamp(x):
        return (x.tanh() + 1.) / 2.

    def unclamp(y):
        z = y * 2. - 1.
        return ((1. + z) / (1. - z)).log() / 2.

    def __init__(self, alpha=0.01, gamma=0.99, optimizer=NoOpOptimizer()):
        self.eps = 1e-8
        parameters = {
            'alpha': torch.sqrt(torch.tensor(alpha)),
            'gamma': RMSProp.unclamp(torch.tensor(gamma))
        }
        ###
        parameter_g_norms = None
        if not isinstance(optimizer, NoOpOptimizer):
            parameter_g_norms = {
                'alpha': [[], 0],
                'gamma': [[], 0]
            }
        ###
        # super().__init__(parameters, optimizer)
        super().__init__(parameters, optimizer, parameter_g_norms)
        self.cache = {}

    # def step(self, params):
    def step(self, params, param_g_norms=None):
        # self.optimizer.step(self.parameters)
        self.optimizer.step(self.parameters, self.parameter_g_norms)
        gamma = RMSProp.clamp(self.parameters['gamma'])
        alpha = torch.square(self.parameters['alpha'])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    's': torch.zeros_like(param)
                }
            g = param.grad.detach()
            ###
            if param_g_norms:
                g = update_param_g_norm_info(param_g_norms, name, g)
            ###
            self.cache[name]['s'] = s = gamma * self.cache[name]['s'].detach() + (1. - gamma) * torch.square(g)
            self.all_params_with_gradients.append(s)
            params[name] = param.detach() - alpha * g / torch.sqrt(s + self.eps)

    def __str__(self):
        return 'rmsprop / ' + str(self.optimizer)

class RMSPropAlpha(Optimizable):
    '''
    A hyperoptimizable RMSProp for only alpha.
    '''
    def __init__(self, alpha=0.01, gamma=0.99, optimizer=NoOpOptimizer()):
        self.eps = 1e-8
        self.gamma = gamma
        parameters = {
            'alpha': torch.sqrt(torch.tensor(alpha)),
        }
        ###
        parameter_g_norms = None
        if not isinstance(optimizer, NoOpOptimizer):
            parameter_g_norms = {
                'alpha': [[], 0]
            }
        ###
        # super().__init__(parameters, optimizer)
        super().__init__(parameters, optimizer, parameter_g_norms)
        self.cache = {}

    # def step(self, params):
    def step(self, params, param_g_norms=None):
        # self.optimizer.step(self.parameters)
        self.optimizer.step(self.parameters, self.parameter_g_norms)
        alpha = torch.square(self.parameters['alpha'])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    's': torch.zeros_like(param)
                }
            g = param.grad.detach()
            ###
            if param_g_norms:
                g = update_param_g_norm_info(param_g_norms, name, g)
            ###
            self.cache[name]['s'] = s = self.gamma * self.cache[name]['s'].detach() + (1. - self.gamma) * torch.square(g)
            self.all_params_with_gradients.append(s)
            params[name] = param.detach() - alpha * g / torch.sqrt(s + self.eps)

    def __str__(self):
        return 'rmspropAlpha / ' + str(self.optimizer)

class Adam(Optimizable):
    '''
    A hyperoptimizable Adam optimizer.
    '''
    def clamp(x):
        return (x.tanh() + 1.) / 2.

    def unclamp(y):
        z = y * 2. - 1.
        return ((1. + z) / (1. - z)).log() / 2.

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, log_eps=-8., optimizer=NoOpOptimizer()):
        self.eps = 10. ** log_eps
        parameters = {
            'alpha': torch.tensor(alpha),
            'beta1': Adam.unclamp(torch.tensor(beta1)),
            'beta2': Adam.unclamp(torch.tensor(beta2)),
        }
        ###
        parameter_g_norms = None
        if not isinstance(optimizer, NoOpOptimizer):
            parameter_g_norms = {
                'alpha': [[], 0],
                'beta1': [[], 0],
                'beta2': [[], 0]
            }
        ###
        # super().__init__(parameters, optimizer)
        super().__init__(parameters, optimizer, parameter_g_norms)
        self.num_stepments = 0
        self.cache = {}

    # def step(self, params):
    def step(self, params, param_g_norms=None):
        self.num_stepments += 1
        # self.optimizer.step(self.parameters)
        self.optimizer.step(self.parameters, self.parameter_g_norms)
        t = self.num_stepments
        beta1 = Adam.clamp(self.parameters['beta1'])
        beta2 = Adam.clamp(self.parameters['beta2'])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param) +\
                            self.eps
# NOTE that we add a little `fudge factor' here because sqrt is not
# differentiable at exactly zero
                }
            g = param.grad.detach()
            self.cache[name]['m'] = m =\
                beta1 * self.cache[name]['m'].detach() + (1. - beta1) * g
            self.cache[name]['v'] = v =\
                beta2 * self.cache[name]['v'].detach() + (1. - beta2) * g * g
            self.all_params_with_gradients.append(m)
            self.all_params_with_gradients.append(v)

            m_hat = m / (1. - beta1 ** float(t))
            v_hat = v / (1. - beta2 ** float(t))

            dparam = m_hat / (v_hat ** 0.5 + self.eps)
            ###
            if param_g_norms:
                g = update_param_g_norm_info(param_g_norms, name, g)
            ###
            params[name] = param.detach() - self.parameters['alpha'] * dparam

    def __str__(self):
        return 'adam / ' + str(self.optimizer)

class AdamBaydin(Optimizable):
    ''' Same as above, but only optimizes the learning rate, treating the
    remaining hyperparameters as constants. '''

    def __init__(
        self,
        alpha=0.001, beta1=0.9, beta2=0.999, log_eps=-8.,
        optimizer=NoOpOptimizer()
    ):
        parameters = {
            'alpha': torch.tensor(alpha),
        }
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.log_eps = log_eps
        ###
        parameter_g_norms = None
        if not isinstance(optimizer, NoOpOptimizer):
            parameter_g_norms = {
                'alpha': [[], 0]
            }
        ###
        # super().__init__(parameters, optimizer)
        super().__init__(parameters, optimizer, parameter_g_norms)
        self.num_stepments = 0
        self.cache = {}

    # def step(self, params):
    def step(self, params, param_g_norms=None):
        self.num_stepments += 1
        # self.optimizer.step(self.parameters)
        self.optimizer.step(self.parameters, self.parameter_g_norms)
        t = self.num_stepments
        beta1 = self.beta1
        beta2 = self.beta2
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param) +\
                            10.**self.log_eps
# NOTE that we add a little `fudge factor' here because sqrt is not
# differentiable at exactly zero
                }

            g = param.grad.detach()
            self.cache[name]['m'] = m =\
                beta1 * self.cache[name]['m'].detach() + (1. - beta1) * g
            self.cache[name]['v'] = v =\
                beta2 * self.cache[name]['v'].detach() + (1. - beta2) * g * g

            self.all_params_with_gradients.append(m)
            self.all_params_with_gradients.append(v)

            m_hat = m / (1. - beta1 ** float(t))
            v_hat = v / (1. - beta2 ** float(t))

            dparam = m_hat / (v_hat ** 0.5 + 10. ** self.log_eps)
            ###
            if param_g_norms:
                g = update_param_g_norm_info(param_g_norms, name, g)
            ###
            params[name] = param.detach() - self.parameters['alpha'] * dparam

    def __str__(self):
        return 'adamBaydin / ' + str(self.optimizer)


class ModuleWrapper(Optimizable):
    '''
    This class tries to convert a torch.nn.Module to an Optimizable, handling
    the internal plumbing needed to update parameters correctly.
    '''
    def __init__(self, module, optimizer=NoOpOptimizer()):
        self.module = module
        parameters = {k:v for k, v in module.named_parameters(recurse=True)}
        super().__init__(parameters, optimizer)

    def initialize(self):
        self.optimizer.initialize()

    def zero_grad(self):
        """ Set all gradients to zero. """
        self.module.zero_grad()
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()

    def forward(self, *xyz):
        return self.module(*xyz)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def step(self):
        self.optimizer.step(self.parameters)
        def set_param(m, k, v):
            kk = k
            while '.' in k:
                sm = k[:k.index('.')]
                k = k[k.index('.') + 1:]
                m = m._modules[sm]

            m._parameters[k] = None
            m._parameters[k] = self.parameters[kk]

        for k, v in self.module.named_parameters(recurse=True):
            set_param(self.module, k, v)

###
def update_param_g_norm_info(param_g_norms, name, g, clip=False):
    g_norm = torch.linalg.vector_norm(g)

    num_g_norms = len(param_g_norms[name][0])
    prev_average_g_norm = param_g_norms[name][1]
    average_g_norm = ((prev_average_g_norm * num_g_norms) + g_norm) / (num_g_norms + 1)

    if clip:
        g *= min(1, average_g_norm / g_norm)
        g_norm = torch.linalg.vector_norm(g)
        average_g_norm = ((prev_average_g_norm * num_g_norms) + g_norm) / (num_g_norms + 1)

    param_g_norms[name][0].append(g_norm)
    param_g_norms[name][1] = average_g_norm

    return g
###