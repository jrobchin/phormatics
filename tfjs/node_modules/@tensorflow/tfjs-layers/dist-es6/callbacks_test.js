var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
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
import { scalar } from '@tensorflow/tfjs-core';
import { BaseLogger, CallbackList, History, resolveScalarsInLogs, disposeTensorsInLogs } from './callbacks';
import { Model } from './engine/training';
import { describeMathCPUAndGPU } from './utils/test_utils';
var MockModel = (function (_super) {
    __extends(MockModel, _super);
    function MockModel(name) {
        return _super.call(this, { inputs: [], outputs: [], name: name }) || this;
    }
    return MockModel;
}(Model));
describe('BaseLogger Callback', function () {
    it('Records and averages losses in an epoch', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var baseLogger, epochLog;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    baseLogger = new BaseLogger();
                    baseLogger.setParams({ metrics: ['loss', 'val_loss'] });
                    return [4, baseLogger.onEpochBegin(0)];
                case 1:
                    _a.sent();
                    return [4, baseLogger.onBatchBegin(0)];
                case 2:
                    _a.sent();
                    return [4, baseLogger.onBatchEnd(0, { batch: 0, size: 10, loss: 5 })];
                case 3:
                    _a.sent();
                    return [4, baseLogger.onBatchBegin(1)];
                case 4:
                    _a.sent();
                    return [4, baseLogger.onBatchEnd(1, { batch: 1, size: 10, loss: 6 })];
                case 5:
                    _a.sent();
                    return [4, baseLogger.onBatchBegin(2)];
                case 6:
                    _a.sent();
                    return [4, baseLogger.onBatchEnd(2, { batch: 2, size: 5, loss: 7 })];
                case 7:
                    _a.sent();
                    epochLog = { val_loss: 3 };
                    return [4, baseLogger.onEpochEnd(0, epochLog)];
                case 8:
                    _a.sent();
                    expect(epochLog['val_loss']).toEqual(3);
                    expect(epochLog['loss'])
                        .toBeCloseTo((10 * 5 + 10 * 6 + 5 * 7) / (10 + 10 + 5));
                    done();
                    return [2];
            }
        });
    }); });
    it('Forgets old epochs', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var baseLogger, numOldEpochs, i, epochLog_1, epochLog;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    baseLogger = new BaseLogger();
                    baseLogger.setParams({ metrics: ['loss', 'val_loss'] });
                    numOldEpochs = 2;
                    i = 0;
                    _a.label = 1;
                case 1:
                    if (!(i < numOldEpochs)) return [3, 7];
                    return [4, baseLogger.onEpochBegin(i)];
                case 2:
                    _a.sent();
                    return [4, baseLogger.onBatchBegin(0)];
                case 3:
                    _a.sent();
                    return [4, baseLogger.onBatchEnd(0, { batch: 0, size: 10, loss: -5 })];
                case 4:
                    _a.sent();
                    epochLog_1 = { val_loss: 3 };
                    return [4, baseLogger.onEpochEnd(i, epochLog_1)];
                case 5:
                    _a.sent();
                    _a.label = 6;
                case 6:
                    ++i;
                    return [3, 1];
                case 7: return [4, baseLogger.onEpochBegin(numOldEpochs)];
                case 8:
                    _a.sent();
                    return [4, baseLogger.onBatchBegin(0)];
                case 9:
                    _a.sent();
                    return [4, baseLogger.onBatchEnd(0, { batch: 0, size: 10, loss: 5 })];
                case 10:
                    _a.sent();
                    return [4, baseLogger.onBatchBegin(1)];
                case 11:
                    _a.sent();
                    return [4, baseLogger.onBatchEnd(1, { batch: 1, size: 10, loss: 6 })];
                case 12:
                    _a.sent();
                    return [4, baseLogger.onBatchBegin(2)];
                case 13:
                    _a.sent();
                    return [4, baseLogger.onBatchEnd(2, { batch: 2, size: 5, loss: 7 })];
                case 14:
                    _a.sent();
                    epochLog = { val_loss: 3 };
                    return [4, baseLogger.onEpochEnd(numOldEpochs, epochLog)];
                case 15:
                    _a.sent();
                    expect(epochLog['val_loss']).toEqual(3);
                    expect(epochLog['loss'])
                        .toBeCloseTo((10 * 5 + 10 * 6 + 5 * 7) / (10 + 10 + 5));
                    done();
                    return [2];
            }
        });
    }); });
});
describe('History Callback', function () {
    it('onTrainBegin', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var history;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    history = new History();
                    return [4, history.onTrainBegin()];
                case 1:
                    _a.sent();
                    expect(history.epoch).toEqual([]);
                    expect(history.history).toEqual({});
                    done();
                    return [2];
            }
        });
    }); });
    it('onEpochEnd', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var history;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    history = new History();
                    return [4, history.onTrainBegin()];
                case 1:
                    _a.sent();
                    return [4, history.onEpochEnd(0, { 'val_loss': 10, 'val_accuracy': 0.1 })];
                case 2:
                    _a.sent();
                    expect(history.epoch).toEqual([0]);
                    expect(history.history).toEqual({ 'val_loss': [10], 'val_accuracy': [0.1] });
                    return [4, history.onEpochEnd(1, { 'val_loss': 9.5, 'val_accuracy': 0.2 })];
                case 3:
                    _a.sent();
                    expect(history.epoch).toEqual([0, 1]);
                    expect(history.history)
                        .toEqual({ 'val_loss': [10, 9.5], 'val_accuracy': [0.1, 0.2] });
                    done();
                    return [2];
            }
        });
    }); });
});
describe('CallbackList', function () {
    it('Constructor with empty arg', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var callbackList;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    callbackList = new CallbackList();
                    return [4, callbackList.onTrainBegin()];
                case 1:
                    _a.sent();
                    return [4, callbackList.onTrainEnd()];
                case 2:
                    _a.sent();
                    done();
                    return [2];
            }
        });
    }); });
    it('Constructor and setParams with array of callbacks', function () {
        var history1 = new History();
        var history2 = new History();
        var callbackList = new CallbackList([history1, history2]);
        var params = { 'verbose': 3 };
        callbackList.setParams(params);
        expect(history1.params).toEqual(params);
        expect(history2.params).toEqual(params);
    });
    it('Constructor and setModel with array of callbacks', function () {
        var history1 = new History();
        var history2 = new History();
        var callbackList = new CallbackList([history1, history2]);
        var model = new MockModel('MockModelA');
        callbackList.setModel(model);
        expect(history1.model).toEqual(model);
        expect(history2.model).toEqual(model);
    });
    it('onTrainBegin', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var history1, history2, callbackList;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    history1 = new History();
                    history2 = new History();
                    callbackList = new CallbackList([history1, history2]);
                    return [4, callbackList.onTrainBegin()];
                case 1:
                    _a.sent();
                    expect(history1.epoch).toEqual([]);
                    expect(history1.history).toEqual({});
                    expect(history2.epoch).toEqual([]);
                    expect(history2.history).toEqual({});
                    done();
                    return [2];
            }
        });
    }); });
    it('onEpochEnd', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var history1, history2, callbackList;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    history1 = new History();
                    history2 = new History();
                    callbackList = new CallbackList([history1, history2]);
                    return [4, callbackList.onTrainBegin()];
                case 1:
                    _a.sent();
                    return [4, callbackList.onEpochEnd(100, { 'val_loss': 10, 'val_accuracy': 0.1 })];
                case 2:
                    _a.sent();
                    expect(history1.epoch).toEqual([100]);
                    expect(history1.history).toEqual({ 'val_loss': [10], 'val_accuracy': [0.1] });
                    expect(history2.epoch).toEqual([100]);
                    expect(history2.history).toEqual({ 'val_loss': [10], 'val_accuracy': [0.1] });
                    return [4, callbackList.onEpochEnd(101, { 'val_loss': 9.5, 'val_accuracy': 0.2 })];
                case 3:
                    _a.sent();
                    expect(history1.epoch).toEqual([100, 101]);
                    expect(history1.history)
                        .toEqual({ 'val_loss': [10, 9.5], 'val_accuracy': [0.1, 0.2] });
                    expect(history2.epoch).toEqual([100, 101]);
                    expect(history2.history)
                        .toEqual({ 'val_loss': [10, 9.5], 'val_accuracy': [0.1, 0.2] });
                    done();
                    return [2];
            }
        });
    }); });
    it('append', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var history1, history2, callbackList;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    history1 = new History();
                    history2 = new History();
                    callbackList = new CallbackList([history1]);
                    return [4, callbackList.onTrainBegin()];
                case 1:
                    _a.sent();
                    expect(history1.epoch).toEqual([]);
                    expect(history1.history).toEqual({});
                    return [4, callbackList.append(history2)];
                case 2:
                    _a.sent();
                    return [4, callbackList.onTrainBegin()];
                case 3:
                    _a.sent();
                    expect(history2.epoch).toEqual([]);
                    expect(history2.history).toEqual({});
                    done();
                    return [2];
            }
        });
    }); });
});
describeMathCPUAndGPU('resolveScalarsInLogs', function () {
    it('Resolve mixed numbers and scalars', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var logs;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    logs = {
                        'a': 1,
                        'b': scalar(2),
                        'c': -3,
                        'd': scalar(-4),
                    };
                    return [4, resolveScalarsInLogs(logs)];
                case 1:
                    _a.sent();
                    expect(logs['a']).toEqual(1);
                    expect(logs['b']).toEqual(2);
                    expect(logs['c']).toEqual(-3);
                    expect(logs['d']).toEqual(-4);
                    done();
                    return [2];
            }
        });
    }); });
    it('Resolve null works fine', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var logs;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    logs = null;
                    return [4, resolveScalarsInLogs(logs)];
                case 1:
                    _a.sent();
                    expect(logs).toEqual(null);
                    done();
                    return [2];
            }
        });
    }); });
    it('Resolve empty works fine', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var logs;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    logs = {};
                    return [4, resolveScalarsInLogs(logs)];
                case 1:
                    _a.sent();
                    expect(logs).toEqual({});
                    done();
                    return [2];
            }
        });
    }); });
});
describeMathCPUAndGPU('disposeTensorsInLogs', function () {
    it('Resolve mixed numbers and scalars', function () {
        var logs = {
            'a': 1,
            'b': scalar(2),
            'c': -3,
            'd': scalar(-4),
        };
        disposeTensorsInLogs(logs);
        expect(logs['a']).toEqual(1);
        expect(logs['b'].isDisposed).toEqual(true);
        expect(logs['c']).toEqual(-3);
        expect(logs['d'].isDisposed).toEqual(true);
    });
});
//# sourceMappingURL=callbacks_test.js.map