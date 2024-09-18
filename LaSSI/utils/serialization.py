import dataclasses
import json


def serialize_configuration(filepath:str, conf):
    if (not isinstance(conf, str)) and (not isinstance(conf, list)):
        w = dataclasses.asdict(conf)
    else:
        w = conf
    with open(str(filepath), 'w') as outfile:
        import yaml
        yaml.dump(w, outfile, default_flow_style=False)


def conf_to_yaml_string(conf):
    import yaml
    yaml.dump(dataclasses.asdict(conf), default_flow_style=False)

def listconf_to_yaml_string(conf):
    import yaml
    return yaml.dump([dataclasses.asdict(x) for x in conf], default_flow_style=False)

def json_dumps(obj):
    from LaSSI.files.JSONDump import EnhancedJSONEncoder
    json.dumps(obj, cls=EnhancedJSONEncoder)