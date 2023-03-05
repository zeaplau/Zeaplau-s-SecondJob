import json
import copy


class ModelConfig(object):
    """Configuration class to store the configuration of a 'Model'
    """
    def __init__(self,
                vocab_size_or_config_json_file,
                hidden_size = 200,
                dropout_prob = 0.1,
                initializer_range= 0.02):

        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.dropout_prob = dropout_prob
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """COnstruct a 'Config' from a Python dictionary of parameters."""
        config = ModelConfig(vocab_size_or_config_json_file = -1)
        for key, value in json_object.items():
            config.__dict__[key]=value
        return config
    @classmethod
    def from_json_file(cls, json_file):
        """Construct a 'Config' from a json file of parameters"""
        with open(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"