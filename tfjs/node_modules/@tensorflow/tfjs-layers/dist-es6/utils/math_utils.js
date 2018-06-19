import * as tfc from '@tensorflow/tfjs-core';
import { scalar, tensor1d } from '@tensorflow/tfjs-core';
import { ValueError } from '../errors';
export function isInteger(x) {
    return x === parseInt(x.toString(), 10);
}
export function arrayProd(array, begin, end) {
    if (begin == null) {
        begin = 0;
    }
    if (end == null) {
        end = array.length;
    }
    var prod = 1;
    for (var i = begin; i < end; ++i) {
        prod *= array[i];
    }
    return prod;
}
function toArray1D(array) {
    array = Array.isArray(array) ? new Float32Array(array) : array;
    return tensor1d(array);
}
export function min(array) {
    return tfc.min(toArray1D(array)).dataSync()[0];
}
export function max(array) {
    return tfc.max(toArray1D(array)).dataSync()[0];
}
export function sum(array) {
    return tfc.sum(toArray1D(array)).dataSync()[0];
}
export function mean(array) {
    return sum(array) / array.length;
}
export function variance(array) {
    var demeaned = tfc.sub(toArray1D(array), scalar(mean(array)));
    var sumSquare = tfc.sum(tfc.mulStrict(demeaned, demeaned)).dataSync()[0];
    return sumSquare / array.length;
}
export function median(array) {
    var arraySorted = array.slice().sort(function (a, b) { return a - b; });
    var lowIdx = Math.floor((arraySorted.length - 1) / 2);
    var highIdx = Math.ceil((arraySorted.length - 1) / 2);
    if (lowIdx === highIdx) {
        return arraySorted[lowIdx];
    }
    return (arraySorted[lowIdx] + arraySorted[highIdx]) / 2;
}
export function range(begin, end) {
    if (end < begin) {
        throw new ValueError("end (" + end + ") < begin (" + begin + ") is forbidden.");
    }
    var out = [];
    for (var i = begin; i < end; ++i) {
        out.push(i);
    }
    return out;
}
//# sourceMappingURL=math_utils.js.map