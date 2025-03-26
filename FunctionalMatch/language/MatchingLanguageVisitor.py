# Generated from MatchingLanguage.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .MatchingLanguageParser import MatchingLanguageParser
else:
    from MatchingLanguageParser import MatchingLanguageParser

# This class defines a complete generic visitor for a parse tree produced by MatchingLanguageParser.

class MatchingLanguageVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by MatchingLanguageParser#language.
    def visitLanguage(self, ctx:MatchingLanguageParser.LanguageContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#function_import.
    def visitFunction_import(self, ctx:MatchingLanguageParser.Function_importContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#class_import.
    def visitClass_import(self, ctx:MatchingLanguageParser.Class_importContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#rule.
    def visitRule(self, ctx:MatchingLanguageParser.RuleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#match_multiobj.
    def visitMatch_multiobj(self, ctx:MatchingLanguageParser.Match_multiobjContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_jpath.
    def visitP_jpath(self, ctx:MatchingLanguageParser.P_jpathContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_lt.
    def visitP_lt(self, ctx:MatchingLanguageParser.P_ltContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_par.
    def visitP_par(self, ctx:MatchingLanguageParser.P_parContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_impl.
    def visitP_impl(self, ctx:MatchingLanguageParser.P_implContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_or.
    def visitP_or(self, ctx:MatchingLanguageParser.P_orContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_gt.
    def visitP_gt(self, ctx:MatchingLanguageParser.P_gtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_call.
    def visitP_call(self, ctx:MatchingLanguageParser.P_callContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_var.
    def visitP_var(self, ctx:MatchingLanguageParser.P_varContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_eq.
    def visitP_eq(self, ctx:MatchingLanguageParser.P_eqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_not.
    def visitP_not(self, ctx:MatchingLanguageParser.P_notContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_geq.
    def visitP_geq(self, ctx:MatchingLanguageParser.P_geqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_isempty.
    def visitP_isempty(self, ctx:MatchingLanguageParser.P_isemptyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_json.
    def visitP_json(self, ctx:MatchingLanguageParser.P_jsonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_neq.
    def visitP_neq(self, ctx:MatchingLanguageParser.P_neqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_and.
    def visitP_and(self, ctx:MatchingLanguageParser.P_andContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#p_leq.
    def visitP_leq(self, ctx:MatchingLanguageParser.P_leqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#extension.
    def visitExtension(self, ctx:MatchingLanguageParser.ExtensionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#actual_object.
    def visitActual_object(self, ctx:MatchingLanguageParser.Actual_objectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#actual_tuple_of_type_and_args.
    def visitActual_tuple_of_type_and_args(self, ctx:MatchingLanguageParser.Actual_tuple_of_type_and_argsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#actual_variable.
    def visitActual_variable(self, ctx:MatchingLanguageParser.Actual_variableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#actual_unary_function_with_args.
    def visitActual_unary_function_with_args(self, ctx:MatchingLanguageParser.Actual_unary_function_with_argsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#ignoring_argument.
    def visitIgnoring_argument(self, ctx:MatchingLanguageParser.Ignoring_argumentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#par.
    def visitPar(self, ctx:MatchingLanguageParser.ParContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#actual_string.
    def visitActual_string(self, ctx:MatchingLanguageParser.Actual_stringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#jpath.
    def visitJpath(self, ctx:MatchingLanguageParser.JpathContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#variable.
    def visitVariable(self, ctx:MatchingLanguageParser.VariableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#rewrite_list.
    def visitRewrite_list(self, ctx:MatchingLanguageParser.Rewrite_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#funarg.
    def visitFunarg(self, ctx:MatchingLanguageParser.FunargContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#replacement.
    def visitReplacement(self, ctx:MatchingLanguageParser.ReplacementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#rewrite.
    def visitRewrite(self, ctx:MatchingLanguageParser.RewriteContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#as.
    def visitAs(self, ctx:MatchingLanguageParser.AsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MatchingLanguageParser#repl.
    def visitRepl(self, ctx:MatchingLanguageParser.ReplContext):
        return self.visitChildren(ctx)



del MatchingLanguageParser