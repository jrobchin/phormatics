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
var AttributeError = (function (_super) {
    __extends(AttributeError, _super);
    function AttributeError(message) {
        var _this = _super.call(this, message) || this;
        Object.setPrototypeOf(_this, AttributeError.prototype);
        return _this;
    }
    return AttributeError;
}(Error));
export { AttributeError };
var RuntimeError = (function (_super) {
    __extends(RuntimeError, _super);
    function RuntimeError(message) {
        var _this = _super.call(this, message) || this;
        Object.setPrototypeOf(_this, RuntimeError.prototype);
        return _this;
    }
    return RuntimeError;
}(Error));
export { RuntimeError };
var ValueError = (function (_super) {
    __extends(ValueError, _super);
    function ValueError(message) {
        var _this = _super.call(this, message) || this;
        Object.setPrototypeOf(_this, ValueError.prototype);
        return _this;
    }
    return ValueError;
}(Error));
export { ValueError };
var NotImplementedError = (function (_super) {
    __extends(NotImplementedError, _super);
    function NotImplementedError(message) {
        var _this = _super.call(this, message) || this;
        Object.setPrototypeOf(_this, NotImplementedError.prototype);
        return _this;
    }
    return NotImplementedError;
}(Error));
export { NotImplementedError };
var AssertionError = (function (_super) {
    __extends(AssertionError, _super);
    function AssertionError(message) {
        var _this = _super.call(this, message) || this;
        Object.setPrototypeOf(_this, AssertionError.prototype);
        return _this;
    }
    return AssertionError;
}(Error));
export { AssertionError };
var IndexError = (function (_super) {
    __extends(IndexError, _super);
    function IndexError(message) {
        var _this = _super.call(this, message) || this;
        Object.setPrototypeOf(_this, IndexError.prototype);
        return _this;
    }
    return IndexError;
}(Error));
export { IndexError };
//# sourceMappingURL=errors.js.map