import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './reduction_executor';
import { createBoolAttr, createNumberAttr, createTensorAttr } from './test_helper';
describe('reduction', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'logical',
            inputNames: ['input1'],
            inputs: [],
            params: {
                x: createTensorAttr(0),
                axis: createNumberAttr(1),
                keepDims: createBoolAttr(true)
            },
            children: []
        };
    });
    describe('executeOp', function () {
        ['max', 'mean', 'min', 'sum'].forEach(function (op) {
            it('should call tfc.' + op, function () {
                var spy = spyOn(tfc, op);
                node.op = op;
                executeOp(node, { input1: input1 }, context);
                expect(spy).toHaveBeenCalledWith(input1[0], 1, true);
            });
        });
        describe('argMax', function () {
            it('should call tfc.argMax', function () {
                spyOn(tfc, 'argMax');
                node.op = 'argMax';
                executeOp(node, { input1: input1 }, context);
                expect(tfc.argMax).toHaveBeenCalledWith(input1[0], 1);
            });
        });
        describe('argMin', function () {
            it('should call tfc.argMin', function () {
                spyOn(tfc, 'argMin');
                node.op = 'argMin';
                executeOp(node, { input1: input1 }, context);
                expect(tfc.argMin).toHaveBeenCalledWith(input1[0], 1);
            });
        });
    });
});
//# sourceMappingURL=reduction_executor_test.js.map