import { tensor2d } from '@tensorflow/tfjs-core';
import * as tfl from '../index';
import { describeMathCPU, describeMathCPUAndGPU, expectTensorsClose } from '../utils/test_utils';
describeMathCPU('leakyReLU: Symbolic', function () {
    it('Correct output shape', function () {
        var layer = tfl.layers.leakyReLU({ alpha: 0.1 });
        var x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
        var y = layer.apply(x);
        expect(y.shape).toEqual(x.shape);
    });
});
describeMathCPUAndGPU('leakyReLU: Tensor', function () {
    it('alpha = default 0.3', function () {
        var layer = tfl.layers.leakyReLU();
        var x = tensor2d([[-1, -2], [0, 3]], [2, 2]);
        var y = layer.apply(x);
        expectTensorsClose(y, tensor2d([[-0.3, -0.6], [0, 3]], [2, 2]));
    });
    it('alpha = 0.1', function () {
        var layer = tfl.layers.leakyReLU({ alpha: 0.1 });
        var x = tensor2d([[-1, -2], [0, 3]], [2, 2]);
        var y = layer.apply(x);
        expectTensorsClose(y, tensor2d([[-0.1, -0.2], [0, 3]], [2, 2]));
    });
});
describeMathCPU('elu: Symbolic', function () {
    it('Correct output shape', function () {
        var layer = tfl.layers.elu();
        var x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
        var y = layer.apply(x);
        expect(y.shape).toEqual(x.shape);
    });
});
describeMathCPUAndGPU('elu: Tensor', function () {
    it('alpha = default 1.0', function () {
        var layer = tfl.layers.elu({});
        var x = tensor2d([[-1, -2], [0, 3]], [2, 2]);
        var y = layer.apply(x);
        expectTensorsClose(y, tensor2d([[Math.exp(-1) - 1, Math.exp(-2) - 1], [0, 3]], [2, 2]));
    });
});
describeMathCPU('thresholdedReLU: Symbolic', function () {
    it('Correct output shape', function () {
        var layer = tfl.layers.thresholdedReLU();
        var x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
        var y = layer.apply(x);
        expect(y.shape).toEqual(x.shape);
    });
});
describeMathCPUAndGPU('thresholdedReLU: Tensor', function () {
    it('theta = default 1.0', function () {
        var layer = tfl.layers.thresholdedReLU({});
        var x = tensor2d([[-1, 0], [1, 3]], [2, 2]);
        var y = layer.apply(x);
        expectTensorsClose(y, tensor2d([[0, 0], [0, 3]], [2, 2]));
    });
});
describeMathCPU('softmax: Symbolic', function () {
    var axisValues = [0, 1, 2, -1, null];
    var _loop_1 = function (axis) {
        it("Correct output shape, axis=" + axis, function () {
            var layer = tfl.layers.softmax({ axis: axis });
            var x = new tfl.SymbolicTensor('float32', [2, 3, 4], null, null, null);
            var y = layer.apply(x);
            expect(y.shape).toEqual(x.shape);
        });
    };
    for (var _i = 0, axisValues_1 = axisValues; _i < axisValues_1.length; _i++) {
        var axis = axisValues_1[_i];
        _loop_1(axis);
    }
});
describeMathCPUAndGPU('softmax: Tensor', function () {
    it('theta = default 1.0', function () {
        var layer = tfl.layers.softmax({});
        var x = tensor2d([[0, 1], [5, 5]], [2, 2]);
        var y = layer.apply(x);
        expectTensorsClose(y, tensor2d([[1 / (1 + Math.E), Math.E / (1 + Math.E)], [0.5, 0.5]], [2, 2]));
    });
});
//# sourceMappingURL=advanced_activation_test.js.map