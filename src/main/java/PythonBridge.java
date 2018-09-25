import org.jpy.PyLib;
import org.jpy.PyModule;
import org.jpy.PyObject;

import java.util.List;
import java.util.Properties;

public class PythonBridge {
    P4 p4;

    public PythonBridge(){
        Properties prop = System.getProperties();

        prop.setProperty("jpy.jpyLib", System.getProperty("user.dir") + "/jpy-build/lib.linux-x86_64-3.6/jpy.cpython-36m-x86_64-linux-gnu.so");
        prop.setProperty("jpy.jdlLib", System.getProperty("user.dir") + "/jpy-build/lib.linux-x86_64-3.6/jdl.cpython-36m-x86_64-linux-gnu.so");
        prop.setProperty("jpy.pythonLib", "/home/trevor/anaconda3/lib/libpython3.6m.so.1.0");

        PyLib.startPython();
        PyModule sys = PyModule.importModule("sys");
        PyObject pathObj = sys.getAttribute("path");

        String executionPath = System.getProperty("user.dir") + "/src/main/python";
        pathObj.call("append", executionPath);

        PyModule.importModule("p3");
        PyModule p4Module = PyModule.importModule("p3.p4");
        PyObject plugInObj = p4Module.call("P4");
        this.p4 = plugInObj.createProxy(P4.class);
    }

    public void start(){
        p4.start();
    }

    public void execute(String[] actions){
        this.p4.execute(actions);
    }

    public float[] getFlatFrame() { return this.p4.get_flat_frame(); }

    public float getReward() { return this.p4.get_frame_reward(); }

    public boolean isPostGame() {return p4.is_post_game(); }

    public void destroy() {
        PyLib.stopPython();
    }
}
