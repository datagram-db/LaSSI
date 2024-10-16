import io
import os.path
from typing import TypeVar, Generic, Callable, Type

T = TypeVar('T')


def target_file_dump(file_name: str,
                     file_load_and_transform: Callable[[io.IOBase], Type[T]],
                     generate_object_from_pipeline: Callable[[], Type[T]],
                     t_to_filestring: Callable[[Type[T]], str],
                     force: bool = True
                     ) -> T:
    if force or (not os.path.isfile(file_name)):
        obj = generate_object_from_pipeline()
        str = t_to_filestring(obj)
        with open(file_name, "w") as f:
            f.write(str)
        return obj
    else:
        with open(file_name, 'r') as f:
            return file_load_and_transform(f)

# def target_folder_dump(file_name,
#                      file_load_and_transform,
#                      processing,
#                      string_transformation,
#                      force=True):
#     if force or (not os.path.isfolder(file_name)):
#         obj = processing()
#         str = string_transformation(obj)
#         with open(file_name, "w") as f:
#             f.write(str)
#         return obj
#     else:
#         with open(file_name, 'r') as f:
#             return file_load_and_transform(f)
