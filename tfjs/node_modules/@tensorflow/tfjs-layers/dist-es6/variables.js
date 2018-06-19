import * as tfc from '@tensorflow/tfjs-core';
import { randomNormal } from './backend/tfjs_backend';
import { getScopedTensorName, getUniqueTensorName } from './common';
import { NotImplementedError } from './errors';
import { getNextUniqueTensorId } from './types';
var DEFAULT_VARIABLE_NAME_PREFIX = 'Variable';
var LayerVariable = (function () {
    function LayerVariable(val, dtype, name, trainable, constraint) {
        if (dtype === void 0) { dtype = 'float32'; }
        if (name === void 0) { name = DEFAULT_VARIABLE_NAME_PREFIX; }
        if (trainable === void 0) { trainable = true; }
        if (constraint === void 0) { constraint = null; }
        this.dtype = dtype == null ? 'float32' : dtype;
        this.shape = val.shape;
        this.id = getNextUniqueTensorId();
        name = name == null ? DEFAULT_VARIABLE_NAME_PREFIX : name;
        this.originalName = getScopedTensorName(name);
        this.name = getUniqueTensorName(this.originalName);
        this.trainable = trainable;
        this.constraint = constraint;
        this.val = tfc.variable(val, this.trainable, this.name, this.dtype);
    }
    LayerVariable.prototype.read = function () {
        return this.val;
    };
    LayerVariable.prototype.write = function (newVal) {
        checkShapesMatch(this.val, newVal);
        this.val.assign(newVal);
        if (this.constraint != null) {
            this.val.assign(this.constraint.apply(this.val));
        }
        return this;
    };
    return LayerVariable;
}());
export { LayerVariable };
function checkShapesMatch(x, y) {
    if (x.shape.toString() !== y.shape.toString()) {
        throw new Error('Shape mismatch: ' + JSON.stringify(x.shape) + ' vs. ' +
            JSON.stringify(y.shape));
    }
}
export function variable(x, dtype, name, constraint) {
    return new LayerVariable(x, dtype, name, true, constraint);
}
export function zerosVariable(shape, dtype, name) {
    return new LayerVariable(tfc.zeros(shape), dtype, name);
}
export function zerosLike(x, dtype, name) {
    return new LayerVariable(tfc.zerosLike(x), dtype, name);
}
export function onesVariable(shape, dtype, name) {
    var allocated = tfc.ones(shape);
    return new LayerVariable(allocated, dtype, name);
}
export function onesLike(x, dtype, name) {
    var allocated = tfc.onesLike(x);
    return new LayerVariable(allocated, dtype, name);
}
export function eyeVariable(size, dtype, name) {
    return new LayerVariable(tfc.eye(size), dtype, name);
}
export function randomUniformVariable(shape, minval, maxval, dtype, seed, name) {
    if (name === void 0) { name = 'randomUniform'; }
    return new LayerVariable(tfc.randomUniform(shape, minval, maxval, dtype), dtype, name);
}
export function truncatedNormalVariable(shape, mean, stddev, dtype, seed, name) {
    if (mean === void 0) { mean = 0.0; }
    if (stddev === void 0) { stddev = 1.0; }
    if (name === void 0) { name = 'truncatedNormal'; }
    if (dtype === 'bool') {
        throw new NotImplementedError("randomNormal does not support dType bool.");
    }
    return new LayerVariable(tfc.truncatedNormal(shape, mean, stddev, dtype, seed), dtype, name);
}
export function randomNormalVariable(shape, mean, stddev, dtype, seed, name) {
    if (mean === void 0) { mean = 0.0; }
    if (stddev === void 0) { stddev = 1.0; }
    if (name === void 0) { name = 'randomNormal'; }
    if (dtype === 'bool') {
        throw new NotImplementedError("randomNormalVariable does not support dType bool.");
    }
    return new LayerVariable(randomNormal(shape, mean, stddev, dtype, seed), dtype, name);
}
export function update(x, xNew) {
    return x.write(xNew);
}
export function updateAdd(x, increment) {
    return x.write(tfc.add(x.read(), increment));
}
export function updateSub(x, decrement) {
    return x.write(tfc.sub(x.read(), decrement));
}
export function batchGetValue(xs) {
    return xs.map(function (x) { return x.read(); });
}
export function batchSetValue(variablesAndValues) {
    variablesAndValues.map(function (variableAndValue) {
        var variable = variableAndValue[0];
        variable.write(variableAndValue[1]);
    });
}
//# sourceMappingURL=variables.js.map