# Generated from MatchingLanguage.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .MatchingLanguageParser import MatchingLanguageParser
else:
    from MatchingLanguageParser import MatchingLanguageParser

# This class defines a complete listener for a parse tree produced by MatchingLanguageParser.
class MatchingLanguageListener(ParseTreeListener):

    # Enter a parse tree produced by MatchingLanguageParser#language.
    def enterLanguage(self, ctx:MatchingLanguageParser.LanguageContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#language.
    def exitLanguage(self, ctx:MatchingLanguageParser.LanguageContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#function_import.
    def enterFunction_import(self, ctx:MatchingLanguageParser.Function_importContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#function_import.
    def exitFunction_import(self, ctx:MatchingLanguageParser.Function_importContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#class_import.
    def enterClass_import(self, ctx:MatchingLanguageParser.Class_importContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#class_import.
    def exitClass_import(self, ctx:MatchingLanguageParser.Class_importContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#rule.
    def enterRule(self, ctx:MatchingLanguageParser.RuleContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#rule.
    def exitRule(self, ctx:MatchingLanguageParser.RuleContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#match_multiobj.
    def enterMatch_multiobj(self, ctx:MatchingLanguageParser.Match_multiobjContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#match_multiobj.
    def exitMatch_multiobj(self, ctx:MatchingLanguageParser.Match_multiobjContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_jpath.
    def enterP_jpath(self, ctx:MatchingLanguageParser.P_jpathContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_jpath.
    def exitP_jpath(self, ctx:MatchingLanguageParser.P_jpathContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_lt.
    def enterP_lt(self, ctx:MatchingLanguageParser.P_ltContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_lt.
    def exitP_lt(self, ctx:MatchingLanguageParser.P_ltContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_par.
    def enterP_par(self, ctx:MatchingLanguageParser.P_parContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_par.
    def exitP_par(self, ctx:MatchingLanguageParser.P_parContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_impl.
    def enterP_impl(self, ctx:MatchingLanguageParser.P_implContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_impl.
    def exitP_impl(self, ctx:MatchingLanguageParser.P_implContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_or.
    def enterP_or(self, ctx:MatchingLanguageParser.P_orContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_or.
    def exitP_or(self, ctx:MatchingLanguageParser.P_orContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_gt.
    def enterP_gt(self, ctx:MatchingLanguageParser.P_gtContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_gt.
    def exitP_gt(self, ctx:MatchingLanguageParser.P_gtContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_call.
    def enterP_call(self, ctx:MatchingLanguageParser.P_callContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_call.
    def exitP_call(self, ctx:MatchingLanguageParser.P_callContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_var.
    def enterP_var(self, ctx:MatchingLanguageParser.P_varContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_var.
    def exitP_var(self, ctx:MatchingLanguageParser.P_varContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_eq.
    def enterP_eq(self, ctx:MatchingLanguageParser.P_eqContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_eq.
    def exitP_eq(self, ctx:MatchingLanguageParser.P_eqContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_not.
    def enterP_not(self, ctx:MatchingLanguageParser.P_notContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_not.
    def exitP_not(self, ctx:MatchingLanguageParser.P_notContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_geq.
    def enterP_geq(self, ctx:MatchingLanguageParser.P_geqContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_geq.
    def exitP_geq(self, ctx:MatchingLanguageParser.P_geqContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_isempty.
    def enterP_isempty(self, ctx:MatchingLanguageParser.P_isemptyContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_isempty.
    def exitP_isempty(self, ctx:MatchingLanguageParser.P_isemptyContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_json.
    def enterP_json(self, ctx:MatchingLanguageParser.P_jsonContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_json.
    def exitP_json(self, ctx:MatchingLanguageParser.P_jsonContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_neq.
    def enterP_neq(self, ctx:MatchingLanguageParser.P_neqContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_neq.
    def exitP_neq(self, ctx:MatchingLanguageParser.P_neqContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_and.
    def enterP_and(self, ctx:MatchingLanguageParser.P_andContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_and.
    def exitP_and(self, ctx:MatchingLanguageParser.P_andContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#p_leq.
    def enterP_leq(self, ctx:MatchingLanguageParser.P_leqContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#p_leq.
    def exitP_leq(self, ctx:MatchingLanguageParser.P_leqContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#extension.
    def enterExtension(self, ctx:MatchingLanguageParser.ExtensionContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#extension.
    def exitExtension(self, ctx:MatchingLanguageParser.ExtensionContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#actual_object.
    def enterActual_object(self, ctx:MatchingLanguageParser.Actual_objectContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#actual_object.
    def exitActual_object(self, ctx:MatchingLanguageParser.Actual_objectContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#actual_tuple_of_type_and_args.
    def enterActual_tuple_of_type_and_args(self, ctx:MatchingLanguageParser.Actual_tuple_of_type_and_argsContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#actual_tuple_of_type_and_args.
    def exitActual_tuple_of_type_and_args(self, ctx:MatchingLanguageParser.Actual_tuple_of_type_and_argsContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#actual_variable.
    def enterActual_variable(self, ctx:MatchingLanguageParser.Actual_variableContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#actual_variable.
    def exitActual_variable(self, ctx:MatchingLanguageParser.Actual_variableContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#actual_unary_function_with_args.
    def enterActual_unary_function_with_args(self, ctx:MatchingLanguageParser.Actual_unary_function_with_argsContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#actual_unary_function_with_args.
    def exitActual_unary_function_with_args(self, ctx:MatchingLanguageParser.Actual_unary_function_with_argsContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#ignoring_argument.
    def enterIgnoring_argument(self, ctx:MatchingLanguageParser.Ignoring_argumentContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#ignoring_argument.
    def exitIgnoring_argument(self, ctx:MatchingLanguageParser.Ignoring_argumentContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#par.
    def enterPar(self, ctx:MatchingLanguageParser.ParContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#par.
    def exitPar(self, ctx:MatchingLanguageParser.ParContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#actual_string.
    def enterActual_string(self, ctx:MatchingLanguageParser.Actual_stringContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#actual_string.
    def exitActual_string(self, ctx:MatchingLanguageParser.Actual_stringContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#jpath.
    def enterJpath(self, ctx:MatchingLanguageParser.JpathContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#jpath.
    def exitJpath(self, ctx:MatchingLanguageParser.JpathContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#variable.
    def enterVariable(self, ctx:MatchingLanguageParser.VariableContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#variable.
    def exitVariable(self, ctx:MatchingLanguageParser.VariableContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#rewrite_list.
    def enterRewrite_list(self, ctx:MatchingLanguageParser.Rewrite_listContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#rewrite_list.
    def exitRewrite_list(self, ctx:MatchingLanguageParser.Rewrite_listContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#funarg.
    def enterFunarg(self, ctx:MatchingLanguageParser.FunargContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#funarg.
    def exitFunarg(self, ctx:MatchingLanguageParser.FunargContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#replacement.
    def enterReplacement(self, ctx:MatchingLanguageParser.ReplacementContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#replacement.
    def exitReplacement(self, ctx:MatchingLanguageParser.ReplacementContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#rewrite.
    def enterRewrite(self, ctx:MatchingLanguageParser.RewriteContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#rewrite.
    def exitRewrite(self, ctx:MatchingLanguageParser.RewriteContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#as.
    def enterAs(self, ctx:MatchingLanguageParser.AsContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#as.
    def exitAs(self, ctx:MatchingLanguageParser.AsContext):
        pass


    # Enter a parse tree produced by MatchingLanguageParser#repl.
    def enterRepl(self, ctx:MatchingLanguageParser.ReplContext):
        pass

    # Exit a parse tree produced by MatchingLanguageParser#repl.
    def exitRepl(self, ctx:MatchingLanguageParser.ReplContext):
        pass



del MatchingLanguageParser