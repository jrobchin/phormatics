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
import { serialization, tidy } from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import { getConstraint, serializeConstraint } from '../constraints';
import { Layer } from '../engine/topology';
import { NotImplementedError, ValueError } from '../errors';
import { getInitializer, serializeInitializer } from '../initializers';
import { getRegularizer, serializeRegularizer } from '../regularizers';
import * as generic_utils from '../utils/generic_utils';
import { getExactlyOneShape } from '../utils/generic_utils';
var Embedding = (function (_super) {
    __extends(Embedding, _super);
    function Embedding(config) {
        var _this = _super.call(this, config) || this;
        _this.embeddings = null;
        _this.DEFAULT_EMBEDDINGS_INITIALIZER = 'randomUniform';
        if (config.batchInputShape == null && config.inputShape == null) {
            var batchSize = null;
            if (config.batchSize != null) {
                batchSize = config.batchSize;
            }
            if (config.inputLength == null) {
                _this.batchInputShape = [batchSize, null];
            }
            else {
                _this.batchInputShape =
                    [batchSize].concat(generic_utils.toList(config.inputLength));
            }
        }
        _this.inputDim = config.inputDim;
        _this.outputDim = config.outputDim;
        _this.embeddingsInitializer = getInitializer(config.embeddingsInitializer || _this.DEFAULT_EMBEDDINGS_INITIALIZER);
        _this.embeddingsRegularizer = getRegularizer(config.embeddingsRegularizer);
        _this.activityRegularizer = getRegularizer(config.activityRegularizer);
        _this.embeddingsConstraint = getConstraint(config.embeddingsConstraint);
        _this.maskZero = config.maskZero;
        _this.inputLength = config.inputLength;
        return _this;
    }
    Embedding.prototype.build = function (inputShape) {
        this.embeddings = this.addWeight('embeddings', [this.inputDim, this.outputDim], this.dtype, this.embeddingsInitializer, this.embeddingsRegularizer, true, this.embeddingsConstraint);
        this.built = true;
    };
    Embedding.prototype.computeMask = function (inputs, mask) {
        throw new NotImplementedError('computeMask has not been implemented for Embedding yet');
    };
    Embedding.prototype.computeOutputShape = function (inputShape) {
        inputShape = generic_utils.getExactlyOneShape(inputShape);
        if (this.inputLength == null) {
            return inputShape.concat([this.outputDim]);
        }
        var inLens = generic_utils.toList(this.inputLength);
        if (inLens.length !== inputShape.length - 1) {
            throw new ValueError("\"inputLength\" is " + this.inputLength + ", but received " +
                ("input shape has shape " + inputShape));
        }
        else {
            var i = 0;
            for (var k = 0; k < inLens.length; ++k) {
                var s1 = inLens[k];
                var s2 = inputShape[k + 1];
                if ((s1 != null) && (s2 != null) && (s1 !== s2)) {
                    throw new ValueError("\"inputLength\" is " + this.inputLength + ", but received " +
                        ("input shape has shape " + inputShape));
                }
                else if (s1 == null) {
                    inLens[i] = s2;
                }
                i++;
            }
        }
        return [inputShape[0]].concat(inLens, [this.outputDim]);
    };
    Embedding.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tidy(function () {
            _this.invokeCallHook(inputs, kwargs);
            var input = generic_utils.getExactlyOneTensor(inputs);
            if (K.dtype(input) !== 'int32') {
                input = K.cast(input, 'int32');
            }
            var output = K.gather(_this.embeddings.read(), input.as1D());
            return output.reshape(getExactlyOneShape(_this.computeOutputShape(input.shape)));
        });
    };
    Embedding.prototype.getConfig = function () {
        var config = {
            inputDim: this.inputDim,
            outputDim: this.outputDim,
            embeddingsInitializer: serializeInitializer(this.embeddingsInitializer),
            embeddingsRegularizer: serializeRegularizer(this.embeddingsRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            embeddingsConstraint: serializeConstraint(this.embeddingsConstraint),
            maskZero: this.maskZero,
            inputLength: this.inputLength
        };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    Embedding.className = 'Embedding';
    return Embedding;
}(Layer));
export { Embedding };
serialization.SerializationMap.register(Embedding);
//# sourceMappingURL=embeddings.js.map