var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './control_executor';
import { createTensorAttr } from './test_helper';
describe('control', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'control',
            inputNames: ['pred', 'input1'],
            inputs: [],
            params: { x: createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('switch', function () {
            it('should set the output condition is true', function () { return __awaiter(_this, void 0, void 0, function () {
                var pred, _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            node.op = 'switch';
                            node.params['pred'] = createTensorAttr(0);
                            node.params['data'] = createTensorAttr(1);
                            pred = [tfc.scalar(true)];
                            _a = expect;
                            return [4, executeOp(node, { pred: pred, input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual([
                                undefined, input1[0]
                            ]);
                            return [2];
                    }
                });
            }); });
            it('should set the output condition is false', function () { return __awaiter(_this, void 0, void 0, function () {
                var pred, _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            node.op = 'switch';
                            node.params['pred'] = createTensorAttr(0);
                            node.params['data'] = createTensorAttr(1);
                            pred = [tfc.scalar(false)];
                            _a = expect;
                            return [4, executeOp(node, { pred: pred, input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual([
                                input1[0], undefined
                            ]);
                            return [2];
                    }
                });
            }); });
        });
        describe('merge', function () {
            it('should return the first available input', function () { return __awaiter(_this, void 0, void 0, function () {
                var pred, _a, _b;
                return __generator(this, function (_c) {
                    switch (_c.label) {
                        case 0:
                            node.op = 'merge';
                            pred = [tfc.scalar(true)];
                            _a = expect;
                            return [4, executeOp(node, { pred: undefined, input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_c.sent()])
                                .toEqual(input1);
                            _b = expect;
                            return [4, executeOp(node, { pred: pred, input1: undefined }, context)];
                        case 2:
                            _b.apply(void 0, [_c.sent()])
                                .toEqual(pred);
                            return [2];
                    }
                });
            }); });
            it('should return undefined if no inputs are available', function () { return __awaiter(_this, void 0, void 0, function () {
                var _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            node.op = 'merge';
                            _a = expect;
                            return [4, executeOp(node, { pred: undefined, input1: undefined }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()])
                                .toEqual(undefined);
                            return [2];
                    }
                });
            }); });
        });
        describe('enter', function () {
            it('should call enterFrame on context', function () { return __awaiter(_this, void 0, void 0, function () {
                var _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            spyOn(context, 'enterFrame');
                            node.op = 'enter';
                            node.params['tensor'] = createTensorAttr(0);
                            node.inputNames = ['input1'];
                            _a = expect;
                            return [4, executeOp(node, { input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual(input1);
                            expect(context.enterFrame).toHaveBeenCalled();
                            return [2];
                    }
                });
            }); });
        });
        describe('exit', function () {
            it('should call existFrame on context', function () { return __awaiter(_this, void 0, void 0, function () {
                var _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            spyOn(context, 'exitFrame');
                            node.op = 'exit';
                            node.params['tensor'] = createTensorAttr(0);
                            node.inputNames = ['input1'];
                            _a = expect;
                            return [4, executeOp(node, { input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual(input1);
                            expect(context.exitFrame).toHaveBeenCalled();
                            return [2];
                    }
                });
            }); });
        });
        describe('nextIteration', function () {
            it('should call nextIteration on context', function () { return __awaiter(_this, void 0, void 0, function () {
                var _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            spyOn(context, 'nextIteration');
                            node.op = 'nextIteration';
                            node.params['tensor'] = createTensorAttr(0);
                            node.inputNames = ['input1'];
                            _a = expect;
                            return [4, executeOp(node, { input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual(input1);
                            expect(context.nextIteration).toHaveBeenCalled();
                            return [2];
                    }
                });
            }); });
        });
    });
});
//# sourceMappingURL=control_executor_test.js.map