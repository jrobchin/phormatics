import { train } from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import { ValueError } from './errors';
export function getOptimizer(identifier) {
    var optimizerMap = {
        'Adagrad': function () { return train.adagrad(.01); },
        'Adam': function () { return train.adam(.001, .9, .999, K.epsilon()); },
        'RMSProp': function () { return train.rmsprop(.001, .9, null, K.epsilon()); },
        'SGD': function () { return train.sgd(.01); }
    };
    optimizerMap['adagrad'] = optimizerMap['Adagrad'];
    optimizerMap['adam'] = optimizerMap['Adam'];
    optimizerMap['rmsprop'] = optimizerMap['RMSProp'];
    optimizerMap['sgd'] = optimizerMap['SGD'];
    if (identifier in optimizerMap) {
        return optimizerMap[identifier]();
    }
    throw new ValueError("Unknown Optimizer " + identifier);
}
//# sourceMappingURL=optimizers.js.map