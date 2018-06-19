import { scalar, tensor1d, tensor2d } from '@tensorflow/tfjs-core';
import * as tfl from './index';
import { binaryAccuracy, categoricalAccuracy, get } from './metrics';
import { describeMathCPUAndGPU, expectTensorsClose } from './utils/test_utils';
describeMathCPUAndGPU('binaryAccuracy', function () {
    it('1D exact', function () {
        var x = tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
        var y = tensor1d([1, 0, 1, 0, 0, 1, 0, 1]);
        var accuracy = tfl.metrics.binaryAccuracy(x, y);
        expectTensorsClose(accuracy, scalar(0.5));
    });
    it('2D thresholded', function () {
        var x = tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
        var y = tensor1d([0.2, 0.4, 0.6, 0.8, 0.2, 0.3, 0.4, 0.7]);
        var accuracy = tfl.metrics.binaryAccuracy(x, y);
        expectTensorsClose(accuracy, scalar(5 / 8));
    });
    it('2D exact', function () {
        var x = tensor2d([[1, 1, 1, 1], [0, 0, 0, 0]], [2, 4]);
        var y = tensor2d([[1, 0, 1, 0], [0, 0, 0, 1]], [2, 4]);
        var accuracy = tfl.metrics.binaryAccuracy(x, y);
        expectTensorsClose(accuracy, tensor1d([0.5, 0.75]));
    });
    it('2D thresholded', function () {
        var x = tensor2d([[1, 1], [1, 1], [0, 0], [0, 0]], [4, 2]);
        var y = tensor2d([[0.2, 0.4], [0.6, 0.8], [0.2, 0.3], [0.4, 0.7]], [4, 2]);
        var accuracy = tfl.metrics.binaryAccuracy(x, y);
        expectTensorsClose(accuracy, tensor1d([0, 1, 1, 0.5]));
    });
});
describeMathCPUAndGPU('binaryCrossentropy', function () {
    it('2D single-value yTrue', function () {
        var x = tensor2d([[0], [0], [0], [1], [1], [1]]);
        var y = tensor2d([[0], [0.5], [1], [0], [0.5], [1]]);
        var accuracy = tfl.metrics.binaryCrossentropy(x, y);
        expectTensorsClose(accuracy, tensor1d([
            1.00000015e-07, 6.93147182e-01, 1.59423847e+01,
            1.61180954e+01, 6.93147182e-01, 1.19209332e-07
        ]));
    });
    it('2D one-hot binary yTrue', function () {
        var x = tensor2d([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);
        var y = tensor2d([[1, 0], [0.5, 0.5], [0, 1], [1, 0], [0.5, 0.5], [0, 1]]);
        var accuracy = tfl.metrics.binaryCrossentropy(x, y);
        expectTensorsClose(accuracy, tensor1d([
            1.0960467e-07, 6.9314718e-01, 1.6030239e+01,
            1.6030239e+01, 6.9314718e-01, 1.0960467e-07
        ]));
    });
});
describeMathCPUAndGPU('categoricalAccuracy', function () {
    it('1D', function () {
        var x = tensor1d([0, 0, 0, 1]);
        var y = tensor1d([0.1, 0.8, 0.05, 0.05]);
        var accuracy = tfl.metrics.categoricalAccuracy(x, y);
        expect(accuracy.dtype).toEqual('float32');
        expect(accuracy.shape).toEqual([]);
        expect(Array.from(accuracy.dataSync())).toEqual([0]);
    });
    it('2D', function () {
        var x = tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]], [2, 4]);
        var y = tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]], [2, 4]);
        var accuracy = tfl.metrics.categoricalAccuracy(x, y);
        expect(accuracy.dtype).toEqual('float32');
        expect(accuracy.shape).toEqual([2]);
        expect(Array.from(accuracy.dataSync())).toEqual([0, 1]);
    });
});
describeMathCPUAndGPU('categoricalCrossentropy metric', function () {
    it('1D', function () {
        var x = tensor1d([0, 0, 0, 1]);
        var y = tensor1d([0.1, 0.8, 0.05, 0.05]);
        var accuracy = tfl.metrics.categoricalCrossentropy(x, y);
        expect(accuracy.dtype).toEqual('float32');
        expectTensorsClose(accuracy, scalar(2.995732));
    });
    it('2D', function () {
        var x = tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]]);
        var y = tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]]);
        var accuracy = tfl.metrics.categoricalCrossentropy(x, y);
        expect(accuracy.dtype).toEqual('float32');
        expectTensorsClose(accuracy, tensor1d([2.995732, 0.22314353]));
    });
});
describe('metrics.get', function () {
    it('valid name, not alias', function () {
        expect(get('binaryAccuracy') === get('categoricalAccuracy')).toEqual(false);
    });
    it('valid name, alias', function () {
        expect(get('mse') === get('MSE')).toEqual(true);
    });
    it('invalid name', function () {
        expect(function () { return get('InvalidMetricName'); }).toThrowError(/Unknown metric/);
    });
    it('LossOrMetricFn input', function () {
        expect(get(binaryAccuracy)).toEqual(binaryAccuracy);
        expect(get(categoricalAccuracy)).toEqual(categoricalAccuracy);
    });
});
//# sourceMappingURL=metrics_test.js.map