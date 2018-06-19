import { ExecutionContext } from './execution_context';
var context;
describe('ExecutionContext', function () {
    beforeEach(function () {
        context = new ExecutionContext({});
    });
    afterEach(function () { });
    it('should initialize', function () {
        expect(context.currentContext).toEqual([
            { id: 0, frameName: '', iterationId: 0 }
        ]);
        expect(context.currentContextId).toEqual('');
    });
    describe('enterFrame', function () {
        it('should add new Frame', function () {
            context.enterFrame('1');
            expect(context.currentContextId).toEqual('/1-0');
            expect(context.currentContext).toEqual([
                { id: 0, frameName: '', iterationId: 0 },
                { id: 1, frameName: '1', iterationId: 0 }
            ]);
        });
    });
    describe('exitFrame', function () {
        it('should remove Frame', function () {
            context.enterFrame('1');
            context.exitFrame();
            expect(context.currentContextId).toEqual('');
            expect(context.currentContext).toEqual([
                { id: 0, frameName: '', iterationId: 0 }
            ]);
        });
        it('should remember previous Frame', function () {
            context.enterFrame('1');
            context.nextIteration();
            context.enterFrame('2');
            context.exitFrame();
            expect(context.currentContextId).toEqual('/1-1');
            expect(context.currentContext).toEqual([
                { id: 0, frameName: '', iterationId: 0 },
                { id: 2, frameName: '1', iterationId: 1 }
            ]);
        });
    });
    describe('nextIteration', function () {
        it('should increate iteration', function () {
            context.enterFrame('1');
            context.nextIteration();
            expect(context.currentContextId).toEqual('/1-1');
            expect(context.currentContext).toEqual([
                { id: 0, frameName: '', iterationId: 0 },
                { id: 2, frameName: '1', iterationId: 1 }
            ]);
        });
    });
});
//# sourceMappingURL=execution_context_test.js.map