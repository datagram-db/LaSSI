# Generated from Parmenides/TBox/parmenides_tbox.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .parmenides_tboxParser import parmenides_tboxParser
else:
    from parmenides_tboxParser import parmenides_tboxParser

# This class defines a complete listener for a parse tree produced by parmenides_tboxParser.
class parmenides_tboxListener(ParseTreeListener):

    # Enter a parse tree produced by parmenides_tboxParser#parmenides_tbox.
    def enterParmenides_tbox(self, ctx:parmenides_tboxParser.Parmenides_tboxContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#parmenides_tbox.
    def exitParmenides_tbox(self, ctx:parmenides_tboxParser.Parmenides_tboxContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#substitutions.
    def enterSubstitutions(self, ctx:parmenides_tboxParser.SubstitutionsContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#substitutions.
    def exitSubstitutions(self, ctx:parmenides_tboxParser.SubstitutionsContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#invention.
    def enterInvention(self, ctx:parmenides_tboxParser.InventionContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#invention.
    def exitInvention(self, ctx:parmenides_tboxParser.InventionContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#fparen.
    def enterFparen(self, ctx:parmenides_tboxParser.FparenContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#fparen.
    def exitFparen(self, ctx:parmenides_tboxParser.FparenContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#rw_variable.
    def enterRw_variable(self, ctx:parmenides_tboxParser.Rw_variableContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#rw_variable.
    def exitRw_variable(self, ctx:parmenides_tboxParser.Rw_variableContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#variable.
    def enterVariable(self, ctx:parmenides_tboxParser.VariableContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#variable.
    def exitVariable(self, ctx:parmenides_tboxParser.VariableContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#unary_predicate.
    def enterUnary_predicate(self, ctx:parmenides_tboxParser.Unary_predicateContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#unary_predicate.
    def exitUnary_predicate(self, ctx:parmenides_tboxParser.Unary_predicateContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#binary_predicate.
    def enterBinary_predicate(self, ctx:parmenides_tboxParser.Binary_predicateContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#binary_predicate.
    def exitBinary_predicate(self, ctx:parmenides_tboxParser.Binary_predicateContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#and.
    def enterAnd(self, ctx:parmenides_tboxParser.AndContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#and.
    def exitAnd(self, ctx:parmenides_tboxParser.AndContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#or.
    def enterOr(self, ctx:parmenides_tboxParser.OrContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#or.
    def exitOr(self, ctx:parmenides_tboxParser.OrContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#not.
    def enterNot(self, ctx:parmenides_tboxParser.NotContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#not.
    def exitNot(self, ctx:parmenides_tboxParser.NotContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#sentence_match.
    def enterSentence_match(self, ctx:parmenides_tboxParser.Sentence_matchContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#sentence_match.
    def exitSentence_match(self, ctx:parmenides_tboxParser.Sentence_matchContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#field_match.
    def enterField_match(self, ctx:parmenides_tboxParser.Field_matchContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#field_match.
    def exitField_match(self, ctx:parmenides_tboxParser.Field_matchContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#data_match_path.
    def enterData_match_path(self, ctx:parmenides_tboxParser.Data_match_pathContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#data_match_path.
    def exitData_match_path(self, ctx:parmenides_tboxParser.Data_match_pathContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#remove.
    def enterRemove(self, ctx:parmenides_tboxParser.RemoveContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#remove.
    def exitRemove(self, ctx:parmenides_tboxParser.RemoveContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#add.
    def enterAdd(self, ctx:parmenides_tboxParser.AddContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#add.
    def exitAdd(self, ctx:parmenides_tboxParser.AddContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#all_properties.
    def enterAll_properties(self, ctx:parmenides_tboxParser.All_propertiesContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#all_properties.
    def exitAll_properties(self, ctx:parmenides_tboxParser.All_propertiesContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#edge_match.
    def enterEdge_match(self, ctx:parmenides_tboxParser.Edge_matchContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#edge_match.
    def exitEdge_match(self, ctx:parmenides_tboxParser.Edge_matchContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#isa_match.
    def enterIsa_match(self, ctx:parmenides_tboxParser.Isa_matchContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#isa_match.
    def exitIsa_match(self, ctx:parmenides_tboxParser.Isa_matchContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#all_queries.
    def enterAll_queries(self, ctx:parmenides_tboxParser.All_queriesContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#all_queries.
    def exitAll_queries(self, ctx:parmenides_tboxParser.All_queriesContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#replacement_pair.
    def enterReplacement_pair(self, ctx:parmenides_tboxParser.Replacement_pairContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#replacement_pair.
    def exitReplacement_pair(self, ctx:parmenides_tboxParser.Replacement_pairContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#key_values.
    def enterKey_values(self, ctx:parmenides_tboxParser.Key_valuesContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#key_values.
    def exitKey_values(self, ctx:parmenides_tboxParser.Key_valuesContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#value.
    def enterValue(self, ctx:parmenides_tboxParser.ValueContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#value.
    def exitValue(self, ctx:parmenides_tboxParser.ValueContext):
        pass


    # Enter a parse tree produced by parmenides_tboxParser#none.
    def enterNone(self, ctx:parmenides_tboxParser.NoneContext):
        pass

    # Exit a parse tree produced by parmenides_tboxParser#none.
    def exitNone(self, ctx:parmenides_tboxParser.NoneContext):
        pass



del parmenides_tboxParser