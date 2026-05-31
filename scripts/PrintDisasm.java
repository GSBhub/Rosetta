// @category Analysis
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.program.model.address.AddressSetView;

public class PrintDisasm extends GhidraScript {
    @Override
    public void run() throws Exception {
        println("========================================");
        println("Language: " + currentProgram.getLanguageID());
        println("Entry:    " + currentProgram.getImageBase());

        FunctionManager fm = currentProgram.getFunctionManager();
        FunctionIterator funcs = fm.getFunctions(true);
        int nFuncs = 0;
        println("\nFunctions:");
        while (funcs.hasNext()) {
            Function fn = funcs.next();
            println("  " + fn.getEntryPoint() + "  " + fn.getName());
            nFuncs++;
        }
        println("Total functions: " + nFuncs);

        println("\nDisassembly:");
        Listing listing = currentProgram.getListing();
        InstructionIterator ii = listing.getInstructions(true);
        int count = 0;
        while (ii.hasNext() && count < 50) {
            Instruction instr = ii.next();
            println("  " + instr.getAddressString(false, true) + "  " + instr.toString());
            count++;
        }
        println("Total instructions: " + count);
        println("========================================");
    }
}
