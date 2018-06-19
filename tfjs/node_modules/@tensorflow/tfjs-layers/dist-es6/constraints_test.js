import { tensor1d } from '@tensorflow/tfjs-core';
import { deserializeConstraint, getConstraint, serializeConstraint } from './constraints';
import * as tfl from './index';
import { describeMathCPU, expectNoLeakedTensors, expectTensorsClose } from './utils/test_utils';
describeMathCPU('Built-in Constraints', function () {
    var initVals;
    beforeEach(function () {
        initVals = tensor1d(new Float32Array([-1, 2, 0, 4, -5, 6]));
    });
    it('NonNeg', function () {
        var constraint = getConstraint('NonNeg');
        var postConstraint = constraint.apply(initVals);
        expectTensorsClose(postConstraint, tensor1d(new Float32Array([0, 2, 0, 4, 0, 6])));
        expectNoLeakedTensors(function () { return constraint.apply(initVals); }, 1);
    });
    it('MaxNorm', function () {
        var constraint = getConstraint('MaxNorm');
        var postConstraint = constraint.apply(initVals);
        expectTensorsClose(postConstraint, tensor1d(new Float32Array([
            -0.2208630521, 0.4417261043, 0, 0.8834522086,
            -1.104315261, 1.325178313
        ])));
        expectNoLeakedTensors(function () { return constraint.apply(initVals); }, 1);
    });
    it('UnitNorm', function () {
        var constraint = getConstraint('UnitNorm');
        var postConstraint = constraint.apply(initVals);
        expectTensorsClose(postConstraint, tensor1d(new Float32Array([
            -0.2208630521 / 2, 0.4417261043 / 2, 0,
            0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
        ])));
        expectNoLeakedTensors(function () { return constraint.apply(initVals); }, 1);
    });
    it('MinMaxNorm', function () {
        var constraint = getConstraint('MinMaxNorm');
        var postConstraint = constraint.apply(initVals);
        expectTensorsClose(postConstraint, tensor1d(new Float32Array([
            -0.2208630521 / 2, 0.4417261043 / 2, 0,
            0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
        ])));
        expectNoLeakedTensors(function () { return constraint.apply(initVals); }, 1);
    });
    it('nonNeg', function () {
        var constraint = getConstraint('nonNeg');
        var postConstraint = constraint.apply(initVals);
        expectTensorsClose(postConstraint, tensor1d(new Float32Array([0, 2, 0, 4, 0, 6])));
    });
    it('maxNorm', function () {
        var constraint = getConstraint('maxNorm');
        var postConstraint = constraint.apply(initVals);
        expectTensorsClose(postConstraint, tensor1d(new Float32Array([
            -0.2208630521, 0.4417261043, 0, 0.8834522086,
            -1.104315261, 1.325178313
        ])));
    });
    it('unitNorm', function () {
        var constraint = getConstraint('unitNorm');
        var postConstraint = constraint.apply(initVals);
        expectTensorsClose(postConstraint, tensor1d(new Float32Array([
            -0.2208630521 / 2, 0.4417261043 / 2, 0,
            0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
        ])));
    });
    it('minMaxNorm', function () {
        var constraint = getConstraint('minMaxNorm');
        var postConstraint = constraint.apply(initVals);
        expectTensorsClose(postConstraint, tensor1d(new Float32Array([
            -0.2208630521 / 2, 0.4417261043 / 2, 0,
            0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
        ])));
    });
});
describeMathCPU('constraints.get', function () {
    it('by string', function () {
        var constraint = getConstraint('maxNorm');
        var config = serializeConstraint(constraint);
        var nestedConfig = config.config;
        expect(nestedConfig.maxValue).toEqual(2);
        expect(nestedConfig.axis).toEqual(0);
    });
    it('by string, upper case', function () {
        var constraint = getConstraint('maxNorm');
        var config = serializeConstraint(constraint);
        var nestedConfig = config.config;
        expect(nestedConfig.maxValue).toEqual(2);
        expect(nestedConfig.axis).toEqual(0);
    });
    it('by existing object', function () {
        var origConstraint = tfl.constraints.nonNeg();
        expect(getConstraint(origConstraint)).toEqual(origConstraint);
    });
    it('by config dict', function () {
        var origConstraint = tfl.constraints.minMaxNorm({ minValue: 0, maxValue: 2, rate: 3, axis: 4 });
        var constraint = getConstraint(serializeConstraint(origConstraint));
        expect(serializeConstraint(constraint))
            .toEqual(serializeConstraint(origConstraint));
    });
});
describe('Constraints Serialization', function () {
    it('Built-ins', function () {
        var constraints = [
            'maxNorm', 'nonNeg', 'unitNorm', 'minMaxNorm', 'MaxNorm', 'NonNeg',
            'UnitNorm', 'MinMaxNorm'
        ];
        for (var _i = 0, constraints_1 = constraints; _i < constraints_1.length; _i++) {
            var name_1 = constraints_1[_i];
            var constraint = getConstraint(name_1);
            var config = serializeConstraint(constraint);
            var reconstituted = deserializeConstraint(config);
            expect(reconstituted).toEqual(constraint);
        }
    });
});
//# sourceMappingURL=constraints_test.js.map