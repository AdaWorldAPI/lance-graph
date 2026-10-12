// Driver authored for oracle capture (not upstream). Delegates to upstream
// org.opennars.core.NALTest.testNAL(path) unchanged, with output echo enabled.
import org.opennars.core.NALTest;

public class RunNal {
    public static void main(String[] args) throws Exception {
        NALTest.showOutput = true;
        NALTest.showSuccess = true;
        int failed = 0;
        for (String p : args) {
            System.out.println("##### FILE " + p);
            try {
                new NALTest(p).testNAL(p);
                System.out.println("##### RESULT PASS " + p);
            } catch (AssertionError e) {
                failed++;
                System.out.println("##### RESULT FAIL " + p + " :: " + e.getMessage());
            }
            System.out.flush(); System.err.flush();
        }
        System.exit(failed == 0 ? 0 : 1);
    }
}
