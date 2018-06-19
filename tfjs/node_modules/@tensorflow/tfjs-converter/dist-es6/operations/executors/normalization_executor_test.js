import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './normalization_executor';
import { createNumberAttr, createTensorAttr } from './test_helper';
describe('normalization', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'normalization',
            inputNames: ['input1'],
            inputs: [],
            params: { x: createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('batchNormalization', function () {
            it('should call tfc.batchNormalization', function () {
                spyOn(tfc, 'batchNormalization');
                node.op = 'batchNormalization';
                node.params.scale = createTensorAttr(1);
                node.params.offset = createTensorAttr(2);
                node.params.mean = createTensorAttr(3);
                node.params.variance = createTensorAttr(4);
                node.params.epsilon = createNumberAttr(5);
                node.inputNames = ['input1', 'input2', 'input3', 'input4', 'input5'];
                var input2 = [tfc.scalar(1)];
                var input3 = [tfc.scalar(2)];
                var input4 = [tfc.scalar(3)];
                var input5 = [tfc.scalar(4)];
                executeOp(node, { input1: input1, input2: input2, input3: input3, input4: input4, input5: input5 }, context);
                expect(tfc.batchNormalization)
                    .toHaveBeenCalledWith(input1[0], input4[0], input5[0], 5, input2[0], input3[0]);
            });
        });
        describe('localResponseNormalization', function () {
            it('should call tfc.localResponseNormalization', function () {
                spyOn(tfc, 'localResponseNormalization');
                node.op = 'localResponseNormalization';
                node.params.radius = createNumberAttr(1);
                node.params.bias = createNumberAttr(2);
                node.params.alpha = createNumberAttr(3);
                node.params.beta = createNumberAttr(4);
                executeOp(node, { input1: input1 }, context);
                expect(tfc.localResponseNormalization)
                    .toHaveBeenCalledWith(input1[0], 1, 2, 3, 4);
            });
        });
        describe('softmax', function () {
            it('should call tfc.softmax', function () {
                spyOn(tfc, 'softmax');
                node.op = 'softmax';
                executeOp(node, { input1: input1 }, context);
                expect(tfc.softmax).toHaveBeenCalledWith(input1[0]);
            });
        });
    });
});
//# sourceMappingURL=normalization_executor_test.js.map