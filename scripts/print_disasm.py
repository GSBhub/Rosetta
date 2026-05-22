# GhidraScript (Jython): print all instructions and functions found.
# Run via: analyzeHeadless ... -postScript print_disasm.py
from ghidra.app.util.headless import HeadlessScript
from ghidra.program.model.listing import CodeUnit

listing = currentProgram.getListing()
func_mgr = currentProgram.getFunctionManager()
lang = currentProgram.getLanguage()

print("=" * 60)
print("Language:", lang.getLanguageID())
print("=" * 60)

# List all functions
functions = list(func_mgr.getFunctions(True))
print(f"\nFunctions found: {len(functions)}")
for fn in functions:
    entry = fn.getEntryPoint()
    print(f"  {fn.getName():<20} @ {entry}")

# Dump all instructions
print("\nDisassembly:")
instr_iter = listing.getInstructions(True)
count = 0
while instr_iter.hasNext() and count < 100:
    instr = instr_iter.next()
    addr = instr.getAddress()
    mnemonic = instr.getMnemonicString()
    ops = instr.getDefaultOperandRepresentationList(0) if instr.getNumOperands() > 0 else []
    print(f"  {addr}  {mnemonic:<10} {instr.toString().split(mnemonic, 1)[-1].strip()}")
    count += 1

print(f"\nTotal instructions disassembled: {count}")
print("=" * 60)
