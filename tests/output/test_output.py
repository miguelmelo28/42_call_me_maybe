from output import Response
from pytest import fixture
import pytest


GOOD_RESPONSES: list[dict] = [
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
      "a": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  }, "prompt":   {
    "prompt": "Calculate the square root of 144"
  }, "parameters": {"a": "144"}},
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  }, "prompt":   {
    "prompt": "Calculate the square root of 144"
  }, "parameters": {"b": "12","a": "144"}},
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "string"
      }
    },
    "returns": {
      "type": "number"
    }
  }, "prompt":   {
    "prompt": "Calculate the square root of 144"
  }, "parameters": {"b": "12","a": "144"}}
]


BAD_RESPONSES: list[dict] = [
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
    "a": {
        "type": "number"
    }
    },
    "returns": {
    "type": "number"
    }
}, "prompt":   {
    "prompt": "Calculate the square root of 144"
}, "parameters": {"a": "144", "b": "132"}},
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
    "a": {
        "type": "string"
    }
    },
    "returns": {
    "type": "number"
    }
}, "prompt":   {
    "prompt": "Calculate the square root of 144"
}, "parameters": {"a": 144}},
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
    "b": {
        "type": "number"
    }
    },
    "returns": {
    "type": "number"
    }
}, "prompt":   {
    "prompt": "Calculate the square root of 144"
}, "parameters": {"a": "144"}},
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
    "a": {
        "type": "number"
    }
    },
    "returns": {
    "type": "number"
    }
}, "prompt":   {
    "prompt": "Calculate the square root of 144"
}, "parameters": {"a": {"hello": "world"}}},
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
    "a": {
        "type": "number"
    }
    },
    "returns": {
    "type": "number"
    }
}, "prompt":   {
    "prompt": "Calculate the square root of 144"
}, "parameters": {"a": "hello"}},
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
    "a": {
        "type": "number"
    }
    },
    "returns": {
    "type": "number"
    }
}, "prompt":   {
    "prompt": "Calculate the square root of 144"
}, "parameters": {"a": "hello"}},
{"name": {
    "nae": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
    "a": {
        "type": "number"
    },
    "b": {
        "type": "number"
    }
    },
    "returns": {
    "type": "number"
    }
}, "prompt":   {
    "prompt": "Calculate the square root of 144"
}, "parameters": {"b": "12","a": "144"}},
{"name": {
    "nae": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
    "a": {
        "type": "number"
    },
    "b": {
        "type": "number"
    }
    },
    "returns": {
    "type": "number"
    }
}, "prompt":   {
    "prompt": "Calculate the square root of 144"
}, "parameters": {"a": "144"}},
{"name": {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "string"
      }
    },
    "returns": {
      "type": "number"
    }
  }, "prompt":   {
    "prompt": "Calculate the square root of 144"
  }, "parameters": {"b": 12,"a": "144"}}
]


@pytest.mark.parametrize("good_response", GOOD_RESPONSES)
# @pytest.mark.skip
def test_valid_response(good_response: dict):
    print(good_response)
    Response.model_validate(good_response)

@pytest.mark.parametrize("bad_response", BAD_RESPONSES)
@pytest.mark.xfail(strict=True)
def test_invalid_response(bad_response: dict):
    try:
        Response.model_validate(bad_response)
    except Exception as e:
        print("\n\n", e)
        raise e
        