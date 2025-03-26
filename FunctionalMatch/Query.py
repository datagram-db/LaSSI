__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2025, Functional Match"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPLv3"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

from collections import defaultdict
from dataclasses import dataclass
from typing import Union

from FunctionalMatch.Match import Match, rewrite_as, evaluate_structural_function
from FunctionalMatch.PropositionalLogic import jpath_update
from FunctionalMatch.ReturningFirstObjects import FromJSONPath, FromVariable, Invent
from FunctionalMatch.TransformationResults import RewriteAs
from FunctionalMatch.utils import FrozenDict


# from FunctionalMatch.functions.Reference import Reference

@dataclass(frozen=True, eq=True, order=True)
class MatchMemo2:
    depth: int
    result_idx: int
    query_id: int
    target_id: int
    json_path:str

@dataclass(frozen=True, eq=True, order=True)
class Query:
    select: Match
    as_: Union[Invent, FromVariable, FromJSONPath, RewriteAs]

    def __call__(self, obj):
        assert isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set)
        obj = list(obj)
        # obj_dict = dict()
        # for x in obj:
        #     from FunctionalMatch.utils import ObjDepthDeterminer
        #     obj_dict[x] = ObjDepthDeterminer.get_depth_dictionary(x)

        test, outcome_list = self.select(obj)
        if not test:
            return test, []  # Rationale: no outcome is available, if the test is false
        results = []

        ## Grouping the outcome list by root objects (info.target_id) and result id (idx)
        # q = self.select.matching_obj_vars
        grouped_results = defaultdict(list)
        for idx, (_, match_preserve_info) in enumerate(outcome_list):
            for x, info in match_preserve_info.items():
                grouped_results[(info.target_id, idx)].append(MatchMemo2(len(info.jsonpath.split(".")), idx, int(x[1:]), info.target_id, info.jsonpath))

        if isinstance(self.as_, RewriteAs):
            for key in grouped_results:
                root_objects, result_id = key
                grouped_results[key] = sorted(grouped_results[key], reverse=not self.as_.isShallow, key=lambda x: x.depth)

            ## Retrieving the original matched value before applying any changes to the data, according to the jpath extracted while matching the object
            orig = dict()
            for key in grouped_results:
                root_objects, result_id = key
                relevant_paths = set()
                for match_memo in grouped_results[key]:
                    dollar_i = match_memo.query_id
                    pi = match_memo.json_path
                    relevant_paths.add(pi)
                from FunctionalMatch.PropositionalLogic import jpath_interpret
                orig[key] = {path: jpath_interpret(obj[root_objects], path) for path in relevant_paths }

            ## Applying the first rewriting process, independent of the object of choice, and applying all the changes
            ## as they were rows. Not considering the nested nature of the data
            rewritten = []
            for dct, match_preserve_info in outcome_list:
                if not isinstance(dct, FrozenDict):
                    assert isinstance(dct, dict)
                    dct = FrozenDict.from_dictionary(dct)
                dct = rewrite_as(dct, self.as_.replacement)
                rewritten.append(dct)

            from FunctionalMatch.example.parmenides.Formulae import FUnaryPredicate
            ## Now, I am updating each object j, recursively, according to the dictionary entry.
            for key in grouped_results:
                root_objects, result_id = key
                curr_obj = obj[root_objects]
                for action in grouped_results[key]:
                    assert action.target_id == root_objects
                    dct = rewritten[action.result_idx]
                    dollar_i = f"${action.query_id}"
                    assert dollar_i in dct
                    dollar_i_obj = dct[dollar_i]
                    j_pi = jpath_interpret(curr_obj, action.json_path)
                    assert action.json_path in orig[key]
                    assert orig[key][action.json_path] is not None
                    if j_pi != orig[key][action.json_path]:
                        # If by effect of any other previous change the values that I am getting do not match anymore with
                        # the previous ones, then I am quitting
                        break
                    if dollar_i_obj == j_pi:
                        ## If the changes by extension altered the $i value, then I am using this as a way to update the
                        ## current object
                        ## Otherwise, if the values are the same, then I am using the dictionary to instantiate the
                        ## object according to the previously matched parameters, that might have been updated, and
                        ## update the object with that instead
                        from FunctionalMatch import instantiate
                        dollar_i_obj = evaluate_structural_function(instantiate(self.select.query[action.query_id], dct))
                    val = jpath_update(curr_obj, action.json_path, dollar_i_obj)
                    assert (not isinstance(val, FUnaryPredicate)) or (val.arg is not None)
                    curr_obj = val
                assert (not isinstance(curr_obj, FUnaryPredicate)) or (curr_obj.arg is not None)
                results.append(curr_obj)
        else:
            for dct, match_preserve_info  in outcome_list:
                result2 = self.as_(dct)
                if result2 is not None:
                    result = evaluate_structural_function(result2.obj)
                    if result is not None:
                        from FunctionalMatch.example.parmenides.Formulae import FUnaryPredicate
                        assert (not isinstance(result, FUnaryPredicate)) or (result.arg is not None)
                        results.append(result)
        return True, results

