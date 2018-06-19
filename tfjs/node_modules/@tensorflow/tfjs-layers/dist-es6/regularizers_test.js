import { scalar, tensor1d } from '@tensorflow/tfjs-core';
import * as tfl from './index';
import { deserializeRegularizer, getRegularizer, serializeRegularizer } from './regularizers';
import { describeMathCPU, expectTensorsClose } from './utils/test_utils';
describeMathCPU('Built-in Regularizers', function () {
    it('l1_l2', function () {
        var x = tensor1d([1, -2, 3, -4]);
        var regularizer = tfl.regularizers.l1l2();
        var score = regularizer.apply(x);
        expectTensorsClose(score, scalar(0.01 * (1 + 2 + 3 + 4) + 0.01 * (1 + 4 + 9 + 16)));
    });
    it('l1', function () {
        var x = tensor1d([1, -2, 3, -4]);
        var regularizer = tfl.regularizers.l1();
        var score = regularizer.apply(x);
        expectTensorsClose(score, scalar(0.01 * (1 + 2 + 3 + 4)));
    });
    it('l2', function () {
        var x = tensor1d([1, -2, 3, -4]);
        var regularizer = tfl.regularizers.l2();
        var score = regularizer.apply(x);
        expectTensorsClose(score, scalar(0.01 * (1 + 4 + 9 + 16)));
    });
    it('l1_l2 non default', function () {
        var x = tensor1d([1, -2, 3, -4]);
        var regularizer = tfl.regularizers.l1l2({ l1: 1, l2: 2 });
        var score = regularizer.apply(x);
        expectTensorsClose(score, scalar(1 * (1 + 2 + 3 + 4) + 2 * (1 + 4 + 9 + 16)));
    });
});
describeMathCPU('regularizers.get', function () {
    var x;
    beforeEach(function () {
        x = tensor1d([1, -2, 3, -4]);
    });
    it('by string - lower camel', function () {
        var regularizer = getRegularizer('l1l2');
        expectTensorsClose(regularizer.apply(x), tfl.regularizers.l1l2().apply(x));
    });
    it('by string - upper camel', function () {
        var regularizer = getRegularizer('L1L2');
        expectTensorsClose(regularizer.apply(x), tfl.regularizers.l1l2().apply(x));
    });
    it('by existing object', function () {
        var origReg = tfl.regularizers.l1l2({ l1: 1, l2: 2 });
        var regularizer = getRegularizer(origReg);
        expect(regularizer).toEqual(origReg);
    });
    it('by config dict', function () {
        var origReg = tfl.regularizers.l1l2({ l1: 1, l2: 2 });
        var regularizer = getRegularizer(serializeRegularizer(origReg));
        expectTensorsClose(regularizer.apply(x), origReg.apply(x));
    });
});
describeMathCPU('Regularizer Serialization', function () {
    it('Built-ins', function () {
        var regularizer = tfl.regularizers.l1l2({ l1: 1, l2: 2 });
        var config = serializeRegularizer(regularizer);
        var reconstituted = deserializeRegularizer(config);
        var roundTripConfig = serializeRegularizer(reconstituted);
        expect(roundTripConfig.className).toEqual('L1L2');
        var nestedConfig = roundTripConfig.config;
        expect(nestedConfig.l1).toEqual(1);
        expect(nestedConfig.l2).toEqual(2);
    });
});
//# sourceMappingURL=regularizers_test.js.map