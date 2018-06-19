import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './matrices_executor';
import { createBoolAttr, createNumericArrayAttr, createTensorAttr } from './test_helper';
describe('matrices', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.scalar(2)];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'matrices',
            inputNames: ['input1', 'input2'],
            inputs: [],
            params: { a: createTensorAttr(0), b: createTensorAttr(1) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('matMul', function () {
            it('should call tfc.matMul', function () {
                spyOn(tfc, 'matMul');
                node.op = 'matMul';
                node.params.transposeA = createBoolAttr(true);
                node.params.transposeB = createBoolAttr(false);
                executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.matMul)
                    .toHaveBeenCalledWith(input1[0], input2[0], true, false);
            });
        });
        describe('transpose', function () {
            it('should call tfc.transpose', function () {
                spyOn(tfc, 'transpose');
                node.op = 'transpose';
                node.inputNames = ['input1', 'input2', 'input3'];
                node.params = {
                    x: createTensorAttr(0),
                    perm: createNumericArrayAttr([1, 2])
                };
                executeOp(node, { input1: input1 }, context);
                expect(tfc.transpose).toHaveBeenCalledWith(input1[0], [1, 2]);
            });
        });
    });
});
//# sourceMappingURL=matrices_executor_test.js.map