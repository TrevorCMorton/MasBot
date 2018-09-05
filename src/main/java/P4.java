import org.jpy.PyObject;

import java.util.List;

public interface P4 {
    PyObject get_frame_reward();
    float[] get_flat_frame();
    void start();
    void execute(String[] action);
}
