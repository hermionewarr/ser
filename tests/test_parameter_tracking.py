from pathlib import Path
import tempfile
from ser.constants import PROJECT_ROOT

from ser.constants import Parameters, save_params, load_params


def test_save_and_load_params():
    """
    The code which loads and saves parameters shouldn't care where it saves to.
    Structuring things that way makes it very simple to test and understand.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        run_path = Path(tmpdir)
        input = Parameters("test", 1, 1, 1.0, "abcdefg", 'today')
        # If we save the parameters to our temporary directory
        save_params(input, run_path)
        # And then load them again
        output = load_params(run_path)
        # We should find that what we loaded is equal to what we saved
        assert input == output


# def test_save_and_load_params():
#     """
#     The code which loads and saves parameters shouldn't care where it saves to.
#     Structuring things that way makes it very simple to test and understand.
#     """
#     input = Parameters("test", 1, 1, 1.0, "abcdefg", 'today')

#     input.make_results_dir()
#     input.make_save_dir()

#     assert (PROJECT_ROOT / "test" / "today" / "model").exists
