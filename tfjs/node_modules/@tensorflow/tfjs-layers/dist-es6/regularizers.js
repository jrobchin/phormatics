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
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import { abs, add, doc, serialization, sum, tidy, zeros } from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import { deserializeKerasObject, serializeKerasObject } from './utils/generic_utils';
var Regularizer = (function (_super) {
    __extends(Regularizer, _super);
    function Regularizer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    return Regularizer;
}(serialization.Serializable));
export { Regularizer };
var L1L2 = (function (_super) {
    __extends(L1L2, _super);
    function L1L2(config) {
        var _this = _super.call(this) || this;
        var l1 = config == null || config.l1 == null ? 0.01 : config.l1;
        var l2 = config == null || config.l2 == null ? 0.01 : config.l2;
        _this.hasL1 = l1 !== 0;
        _this.hasL2 = l2 !== 0;
        _this.l1 = K.getScalar(l1);
        _this.l2 = K.getScalar(l2);
        return _this;
    }
    L1L2.prototype.apply = function (x) {
        var _this = this;
        return tidy(function () {
            var regularization = zeros([1]);
            if (_this.hasL1) {
                regularization =
                    add(regularization, sum(K.scalarTimesArray(_this.l1, abs(x))));
            }
            if (_this.hasL2) {
                regularization =
                    add(regularization, sum(K.scalarTimesArray(_this.l2, K.square(x))));
            }
            return regularization.asScalar();
        });
    };
    L1L2.prototype.getConfig = function () {
        return { 'l1': this.l1.dataSync()[0], 'l2': this.l2.dataSync()[0] };
    };
    L1L2.fromConfig = function (cls, config) {
        return new cls({ l1: config.l1, l2: config.l2 });
    };
    L1L2.className = 'L1L2';
    L1L2 = __decorate([
        doc({ heading: 'Regularizers', namespace: 'regularizers' })
    ], L1L2);
    return L1L2;
}(Regularizer));
export { L1L2 };
serialization.SerializationMap.register(L1L2);
export function l1(config) {
    return new L1L2({ l1: config != null ? config.l1 : null, l2: 0 });
}
export function l2(config) {
    return new L1L2({ l2: config != null ? config.l2 : null, l1: 0 });
}
export var REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP = {
    'l1l2': 'L1L2'
};
export function serializeRegularizer(constraint) {
    return serializeKerasObject(constraint);
}
export function deserializeRegularizer(config, customObjects) {
    if (customObjects === void 0) { customObjects = {}; }
    return deserializeKerasObject(config, serialization.SerializationMap.getMap().classNameMap, customObjects, 'regularizer');
}
export function getRegularizer(identifier) {
    if (identifier == null) {
        return null;
    }
    if (typeof identifier === 'string') {
        var className = identifier in REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
            REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
            identifier;
        var config = { className: className, config: {} };
        return deserializeRegularizer(config);
    }
    else if (identifier instanceof Regularizer) {
        return identifier;
    }
    else {
        return deserializeRegularizer(identifier);
    }
}
//# sourceMappingURL=regularizers.js.map