var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
import { serialization, tidy, util } from '@tensorflow/tfjs-core';
import { getActivation, serializeActivation } from '../activations';
import * as K from '../backend/tfjs_backend';
import { getConstraint, serializeConstraint } from '../constraints';
import { Layer } from '../engine/topology';
import { NotImplementedError, ValueError } from '../errors';
import { getInitializer, serializeInitializer } from '../initializers';
import { getRegularizer, serializeRegularizer } from '../regularizers';
import * as generic_utils from '../utils/generic_utils';
import { getExactlyOneTensor } from '../utils/generic_utils';
import * as math_utils from '../utils/math_utils';
var Dropout = (function (_super) {
    __extends(Dropout, _super);
    function Dropout(config) {
        var _this = _super.call(this, config) || this;
        _this.rate = Math.max(Math.min(config.rate, 1), 0);
        _this.rateScalar = K.getScalar(_this.rate);
        _this.noiseShape = config.noiseShape;
        _this.seed = config.seed;
        if (_this.seed != null) {
            throw new NotImplementedError('Non-default seed is not implemented in Dropout layer yet: ' +
                _this.seed);
        }
        _this.supportsMasking = true;
        return _this;
    }
    Dropout.prototype.getNoiseShape = function (input) {
        if (this.noiseShape == null) {
            return this.noiseShape;
        }
        var inputShape = input.shape;
        var noiseShape = [];
        for (var i = 0; i < this.noiseShape.length; ++i) {
            noiseShape.push(this.noiseShape[i] == null ? inputShape[i] : this.noiseShape[i]);
        }
        return noiseShape;
    };
    Dropout.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            _this.invokeCallHook(inputs, kwargs);
            var input = generic_utils.getExactlyOneTensor(inputs);
            if (_this.noiseShape != null &&
                !util.arraysEqual(input.shape, _this.noiseShape)) {
                throw new NotImplementedError('Non-default noise shape is not implemented in Dropout ' +
                    'layer yet: ' + JSON.stringify(_this.noiseShape));
            }
            if (0 < _this.rate && _this.rate < 1) {
                var training = kwargs['training'] == null ? false : kwargs['training'];
                var noiseShape_1 = _this.getNoiseShape(input);
                var output = K.inTrainPhase(function () { return K.dropout(input, _this.rateScalar, noiseShape_1, _this.seed); }, function () { return input; }, training);
                return output;
            }
            return inputs;
        });
    };
    Dropout.prototype.getConfig = function () {
        var config = {
            rate: this.rate,
            noiseShape: this.noiseShape,
            seed: this.seed,
        };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    Dropout.className = 'Dropout';
    return Dropout;
}(Layer));
export { Dropout };
serialization.SerializationMap.register(Dropout);
var Dense = (function (_super) {
    __extends(Dense, _super);
    function Dense(config) {
        var _this = _super.call(this, config) || this;
        _this.activation = null;
        _this.useBias = true;
        _this.kernel = null;
        _this.bias = null;
        _this.DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
        _this.DEFAULT_BIAS_INITIALIZER = 'zeros';
        if (config.batchInputShape == null && config.inputShape == null &&
            config.inputDim != null) {
            var batchSize = null;
            if (config.batchSize != null) {
                batchSize = config.batchSize;
            }
            _this.batchInputShape = [batchSize, config.inputDim];
        }
        _this.units = config.units;
        _this.activation = getActivation(config.activation);
        if (config.useBias != null) {
            _this.useBias = config.useBias;
        }
        _this.kernelInitializer = getInitializer(config.kernelInitializer || _this.DEFAULT_KERNEL_INITIALIZER);
        _this.biasInitializer =
            getInitializer(config.biasInitializer || _this.DEFAULT_BIAS_INITIALIZER);
        _this.kernelConstraint = getConstraint(config.kernelConstraint);
        _this.biasConstraint = getConstraint(config.biasConstraint);
        _this.kernelRegularizer = getRegularizer(config.kernelRegularizer);
        _this.biasRegularizer = getRegularizer(config.biasRegularizer);
        _this.activityRegularizer = getRegularizer(config.activityRegularizer);
        _this.inputSpec = [{ minNDim: 2 }];
        return _this;
    }
    Dense.prototype.build = function (inputShape) {
        inputShape = generic_utils.getExactlyOneShape(inputShape);
        var inputLastDim = inputShape[inputShape.length - 1];
        if (this.kernel == null) {
            this.kernel = this.addWeight('kernel', [inputLastDim, this.units], null, this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
            if (this.useBias) {
                this.bias = this.addWeight('bias', [this.units], null, this.biasInitializer, this.biasRegularizer, true, this.biasConstraint);
            }
        }
        this.inputSpec = [{ minNDim: 2, axes: (_a = {}, _a[-1] = inputLastDim, _a) }];
        this.built = true;
        var _a;
    };
    Dense.prototype.computeOutputShape = function (inputShape) {
        inputShape = generic_utils.getExactlyOneShape(inputShape);
        var outputShape = inputShape.slice();
        outputShape[outputShape.length - 1] = this.units;
        return outputShape;
    };
    Dense.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            _this.invokeCallHook(inputs, kwargs);
            var input = generic_utils.getExactlyOneTensor(inputs);
            var output = K.dot(input, _this.kernel.read());
            if (_this.bias != null) {
                output = K.biasAdd(output, _this.bias.read());
            }
            if (_this.activation != null) {
                output = _this.activation.apply(output);
            }
            return output;
        });
    };
    Dense.prototype.getConfig = function () {
        var config = {
            units: this.units,
            activation: serializeActivation(this.activation),
            useBias: this.useBias,
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            biasInitializer: serializeInitializer(this.biasInitializer),
            kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
            biasRegularizer: serializeRegularizer(this.biasRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            kernelConstraint: serializeConstraint(this.kernelConstraint),
            biasConstraint: serializeConstraint(this.biasConstraint)
        };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    Dense.className = 'Dense';
    return Dense;
}(Layer));
export { Dense };
serialization.SerializationMap.register(Dense);
var Flatten = (function (_super) {
    __extends(Flatten, _super);
    function Flatten(config) {
        var _this = _super.call(this, config || {}) || this;
        _this.inputSpec = [{ minNDim: 3 }];
        return _this;
    }
    Flatten.prototype.computeOutputShape = function (inputShape) {
        inputShape = generic_utils.getExactlyOneShape(inputShape);
        for (var _i = 0, _a = inputShape.slice(1); _i < _a.length; _i++) {
            var dim = _a[_i];
            if (dim == null) {
                throw new ValueError("The shape of the input to \"Flatten\" is not fully defined " +
                    ("(got " + inputShape.slice(1) + "). Make sure to pass a complete ") +
                    "\"input_shape\" or \"batch_input_shape\" argument to the first " +
                    "layer in your model.");
            }
        }
        return [inputShape[0], math_utils.arrayProd(inputShape, 1)];
    };
    Flatten.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            _this.invokeCallHook(inputs, kwargs);
            return K.batchFlatten(generic_utils.getExactlyOneTensor(inputs));
        });
    };
    Flatten.className = 'Flatten';
    return Flatten;
}(Layer));
export { Flatten };
serialization.SerializationMap.register(Flatten);
var Activation = (function (_super) {
    __extends(Activation, _super);
    function Activation(config) {
        var _this = _super.call(this, config) || this;
        _this.supportsMasking = true;
        _this.activation = getActivation(config.activation);
        return _this;
    }
    Activation.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            _this.invokeCallHook(inputs, kwargs);
            var input = generic_utils.getExactlyOneTensor(inputs);
            return _this.activation.apply(input);
        });
    };
    Activation.prototype.getConfig = function () {
        var config = { activation: serializeActivation(this.activation) };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    Activation.className = 'Activation';
    return Activation;
}(Layer));
export { Activation };
serialization.SerializationMap.register(Activation);
var RepeatVector = (function (_super) {
    __extends(RepeatVector, _super);
    function RepeatVector(config) {
        var _this = _super.call(this, config) || this;
        _this.n = config.n;
        _this.inputSpec = [{ ndim: 2 }];
        return _this;
    }
    RepeatVector.prototype.computeOutputShape = function (inputShape) {
        return [inputShape[0], this.n, inputShape[1]];
    };
    RepeatVector.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            inputs = getExactlyOneTensor(inputs);
            return K.repeat(inputs, _this.n);
        });
    };
    RepeatVector.prototype.getConfig = function () {
        var config = {
            n: this.n,
        };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    RepeatVector.className = 'RepeatVector';
    return RepeatVector;
}(Layer));
export { RepeatVector };
serialization.SerializationMap.register(RepeatVector);
var Reshape = (function (_super) {
    __extends(Reshape, _super);
    function Reshape(config) {
        var _this = _super.call(this, config) || this;
        _this.targetShape = config.targetShape;
        for (var i = 0; i < _this.targetShape.length; ++i) {
            if (_this.isUnknown(_this.targetShape[i])) {
                _this.targetShape[i] = null;
            }
        }
        return _this;
    }
    Reshape.prototype.isUnknown = function (dim) {
        return dim < 0 || dim == null;
    };
    Reshape.prototype.fixUnknownDimension = function (inputShape, outputShape) {
        var errorMsg = 'Total size of new array must be unchanged.';
        var finalShape = outputShape.slice();
        var known = 1;
        var unknown = null;
        for (var i = 0; i < finalShape.length; ++i) {
            var dim = finalShape[i];
            if (this.isUnknown(dim)) {
                if (unknown === null) {
                    unknown = i;
                }
                else {
                    throw new ValueError('Can only specifiy one unknown dimension.');
                }
            }
            else {
                known *= dim;
            }
        }
        var originalSize = math_utils.arrayProd(inputShape);
        if (unknown !== null) {
            if (known === 0 || originalSize % known !== 0) {
                throw new ValueError(errorMsg);
            }
            finalShape[unknown] = originalSize / known;
        }
        else if (originalSize !== known) {
            throw new ValueError(errorMsg);
        }
        return finalShape;
    };
    Reshape.prototype.computeOutputShape = function (inputShape) {
        var anyUnknownDims = false;
        for (var i = 0; i < inputShape.length; ++i) {
            if (this.isUnknown(inputShape[i])) {
                anyUnknownDims = true;
                break;
            }
        }
        if (anyUnknownDims) {
            return inputShape.slice(0, 1).concat(this.targetShape);
        }
        else {
            return inputShape.slice(0, 1).concat(this.fixUnknownDimension(inputShape.slice(1), this.targetShape));
        }
    };
    Reshape.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            _this.invokeCallHook(inputs, kwargs);
            var input = generic_utils.getExactlyOneTensor(inputs);
            var inputShape = K.shape(input);
            var outputShape = inputShape.slice(0, 1).concat(_this.fixUnknownDimension(inputShape.slice(1), _this.targetShape));
            return input.reshape(outputShape);
        });
    };
    Reshape.prototype.getConfig = function () {
        var config = {
            targetShape: this.targetShape,
        };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    Reshape.className = 'Reshape';
    return Reshape;
}(Layer));
export { Reshape };
serialization.SerializationMap.register(Reshape);
//# sourceMappingURL=core.js.map