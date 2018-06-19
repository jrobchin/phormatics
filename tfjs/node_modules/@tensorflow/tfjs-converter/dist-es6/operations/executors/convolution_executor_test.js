import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './convolution_executor';
import { createNumberAttr, createNumericArrayAttr, createStrAttr, createTensorAttr } from './test_helper';
describe('convolution', function () {
    var node;
    var input = [tfc.scalar(1)];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'convolution',
            inputNames: ['input'],
            inputs: [],
            params: { x: createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('avgPool', function () {
            it('should call tfc.avgPool', function () {
                spyOn(tfc, 'avgPool');
                node.op = 'avgPool';
                node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = createStrAttr('same');
                node.params['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);
                executeOp(node, { input: input }, context);
                expect(tfc.avgPool)
                    .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
            });
        });
        describe('maxPool', function () {
            it('should call tfc.maxPool', function () {
                spyOn(tfc, 'maxPool');
                node.op = 'maxPool';
                node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = createStrAttr('same');
                node.params['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);
                executeOp(node, { input: input }, context);
                expect(tfc.maxPool)
                    .toHaveBeenCalledWith(input[0], [2, 2], [2, 2], 'same');
            });
        });
        describe('Conv2d', function () {
            it('should call tfc.conv2d', function () {
                spyOn(tfc, 'conv2d');
                node.op = 'conv2d';
                node.params['filter'] = createTensorAttr(1);
                node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = createStrAttr('same');
                node.params['dataFormat'] = createStrAttr('NHWC');
                node.params['dilations'] = createNumericArrayAttr([2, 2]);
                var input1 = [tfc.scalar(1.0)];
                var input2 = [tfc.scalar(1.0)];
                node.inputNames = ['input1', 'input2'];
                executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.conv2d)
                    .toHaveBeenCalledWith(input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
            });
        });
        describe('conv2dTranspose', function () {
            it('should call tfc.conv2dTranspose', function () {
                spyOn(tfc, 'conv2dTranspose');
                node.op = 'conv2dTranspose';
                node.params['outputShape'] = createNumericArrayAttr([1, 2, 2, 2]);
                node.params['filter'] = createTensorAttr(1);
                node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = createStrAttr('same');
                var input1 = [tfc.scalar(1.0)];
                var input2 = [tfc.scalar(1.0)];
                node.inputNames = ['input1', 'input2'];
                executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.conv2dTranspose)
                    .toHaveBeenCalledWith(input1[0], input2[0], [1, 2, 2, 2], [2, 2], 'same');
            });
        });
        describe('Conv1d', function () {
            it('should call tfc.conv1d', function () {
                spyOn(tfc, 'conv1d');
                node.op = 'conv1d';
                node.category = 'convolution';
                node.params['filter'] = createTensorAttr(1);
                node.params['stride'] = createNumberAttr(1);
                node.params['pad'] = createStrAttr('same');
                node.params['dataFormat'] = createStrAttr('NWC');
                node.params['dilation'] = createNumberAttr(1);
                var input1 = [tfc.scalar(1.0)];
                var input2 = [tfc.scalar(1.0)];
                node.inputNames = ['input1', 'input2'];
                executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.conv1d)
                    .toHaveBeenCalledWith(input1[0], input2[0], 1, 'same', 'NWC', 1);
            });
        });
        describe('depthwiseConv2d', function () {
            it('should call tfc.depthwiseConv2d', function () {
                spyOn(tfc, 'depthwiseConv2d');
                node.op = 'depthwiseConv2d';
                node.category = 'convolution';
                node.params['input'] = createTensorAttr(0);
                node.params['filter'] = createTensorAttr(1);
                node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
                node.params['pad'] = createStrAttr('same');
                node.params['dataFormat'] = createStrAttr('NHWC');
                node.params['dilations'] = createNumericArrayAttr([2, 2]);
                var input1 = [tfc.scalar(1.0)];
                var input2 = [tfc.scalar(1.0)];
                node.inputNames = ['input1', 'input2'];
                executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.depthwiseConv2d)
                    .toHaveBeenCalledWith(input1[0], input2[0], [2, 2], 'same', 'NHWC', [2, 2]);
            });
        });
    });
});
//# sourceMappingURL=convolution_executor_test.js.map