import os.path

def target_file_dump(file_name,
                     file_load_and_transform,
                     processing,
                     string_transformation,
                     force=True):
    if force or (not os.path.isfile(file_name)):
        obj = processing()
        str = string_transformation(obj)
        with open(file_name, "w") as f:
            f.write(str)
        return obj
    else:
        with open(file_name, 'r') as f:
            return file_load_and_transform(f)
