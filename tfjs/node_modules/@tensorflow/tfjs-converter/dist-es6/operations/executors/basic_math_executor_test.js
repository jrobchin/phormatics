import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './basic_math_executor';
import { createNumberAttr, createTensorAttr } from './test_helper';
describe('basic math', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'basic_math',
            inputNames: ['input1'],
            inputs: [],
            params: { x: createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        ['abs', 'acos', 'asin', 'atan', 'ceil', 'cos', 'cosh', 'elu', 'exp',
            'floor', 'log', 'neg', 'relu', 'selu', 'sigmoid', 'sin', 'sinh', 'sqrt',
            'square', 'tanh', 'tan', 'sign', 'round', 'expm1', 'log1p', 'reciprocal',
            'softplus', 'asinh', 'acosh', 'atanh', 'erf']
            .forEach(function (op) {
            it('should call tfc.' + op, function () {
                var spy = spyOn(tfc, op);
                node.op = op;
                executeOp(node, { input1: input1 }, context);
                expect(spy).toHaveBeenCalledWith(input1[0]);
            });
        });
        describe('clipByValue', function () {
            it('should call tfc.clipByValue', function () {
                spyOn(tfc, 'clipByValue');
                node.op = 'clipByValue';
                node.params['clipValueMax'] = createNumberAttr(6);
                node.params['clipValueMin'] = createNumberAttr(0);
                executeOp(node, { input1: input1 }, context);
                expect(tfc.clipByValue).toHaveBeenCalledWith(input1[0], 0, 6);
            });
        });
        describe('rsqrt', function () {
            it('should call tfc.div', function () {
                var input1 = [tfc.scalar(1)];
                node.op = 'rsqrt';
                spyOn(tfc, 'div');
                spyOn(tfc, 'sqrt').and.returnValue(input1);
                executeOp(node, { input1: input1 }, context);
                expect(tfc.sqrt).toHaveBeenCalledWith(input1[0]);
                expect(tfc.div).toHaveBeenCalledWith(jasmine.any(tfc.Tensor), input1);
            });
        });
    });
});
//# sourceMappingURL=basic_math_executor_test.js.map