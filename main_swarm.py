import os
import sqlite3
import subprocess
import z3
import sys
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- 1. INITIALIZATION & SETUP ---
load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY is missing from the .env file.")

# Initialize the LLM
# --- 2. OPENROUTER LLM SETUP ---
# --- 2. OPENROUTER LLM SETUP ---
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="arcee-ai/trinity-large-preview:free",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    # Add these two lines to handle the provider flakiness:
    timeout=120,    # Don't wait 5 mins; if it takes > 2 mins, it's probably hung
    max_retries=3   # Automatically try again if the provider times out
)

# --- 2. STATE DEFINITION ---
class SwarmState(TypedDict):
    iteration_count: int
    math_string: str          # The SMT-LIB2 logic string for Z3
    current_error_log: str    # Passed between agents to learn from mistakes
    success: bool             # The kill-switch for the 24/7 loop

# --- 3. HARD-CODED LOCAL TOOLS (The Access Matrix) ---

@tool
def write_theory(theory_text: str, smt_logic: str) -> str:
    """
    AGENT ALPHA USE ONLY.
    Overwrites 1_theory.txt with new mathematical logic and natural language explanations.
    Requires an SMT-LIB2 logic string to be passed for Z3 verification.
    """
    try:
        with open("1_theory.txt", "w", encoding="utf-8") as f:
            f.write(theory_text)
        return smt_logic # Passed back to update LangGraph state
    except Exception as e:
        return f"File Write Error: {str(e)}"

@tool
def write_engine(code: str) -> str:
    """
    AGENT BETA USE ONLY.
    Overwrites 2_engine.py with optimized, dependency-free NumPy/C-Type code.
    """
    try:
        with open("2_engine.py", "w", encoding="utf-8") as f:
            f.write(code)
        return "Engine file updated successfully."
    except Exception as e:
        return f"File Write Error: {str(e)}"

@tool
def run_z3_proof(smt2_string: str) -> str:
    """
    AGENT GAMMA USE ONLY.
    Executes a Z3 theorem prover instance in-memory using the provided SMT-LIB2 string.
    Returns 'sat' (satisfiable/provable), 'unsat', or syntax errors.
    """
    if not smt2_string.strip():
        return "Error: No SMT-LIB2 logic provided."
    try:
        s = z3.Solver()
        s.from_string(smt2_string)
        result = s.check()
        return f"Z3 Result: {result}"
    except Exception as e:
        return f"Z3 Syntax/Logic Error: {str(e)}"

@tool
def execute_benchmark() -> str:
    """
    AGENT GAMMA USE ONLY.
    Executes 'python 3_benchmark.py'. If it fails, captures stderr. If it passes, captures stdout.
    """
    try:
        result = subprocess.run(
            [sys.executable, "3_benchmark.py"], 
            capture_output=True, 
            text=True, 
            timeout=120 # Kill if Beta writes an infinite loop
        )
        if result.returncode == 0:
            return f"SUCCESS:\n{result.stdout}"
        else:
            return f"FAILED:\n{result.stderr}\n{result.stdout}"
    except subprocess.TimeoutExpired:
        return "FAILED: Timeout execution exceeded 120 seconds. Code likely contains an infinite loop or horrific memory leak."
    except Exception as e:
        return f"FAILED: System Error: {str(e)}"

@tool
def append_critique(error_log: str) -> str:
    """
    AGENT GAMMA USE ONLY.
    Appends a failure log to the bottom of 1_theory.txt so Agent Alpha reads it on the next loop.
    """
    try:
        with open("1_theory.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n=== GAMMA CRITIQUE (ITERATION FAILED) ===\n{error_log}\n")
        return "Critique appended."
    except Exception as e:
        return f"File Write Error: {str(e)}"

# --- 4. AGENT LOGIC NODES ---

def agent_alpha_node(state: SwarmState):
    print(f"\n[ITERATION {state['iteration_count']}] -> AWAKENING AGENT ALPHA (Theorist)...")
    
    prompt = f"""[DIRECTIVE: HEADLESS MODE]
    You are an autonomous mathematical research agent. 
    TASK: Invent a new neural network training algorithm replacing backpropagation.
    
    CONSTRAINTS:
    1. Forward-pass only / Local-error / Synthetic gradients.
    2. Zero standard backprop.
    3. Must be Z3-provable.
    
    CURRENT FAILURE LOG:
    {state.get('current_error_log', 'Initial Run')}

    OUTPUT RULES:
    - DO NOT provide introductory text or pleasantries.
    - DO NOT explain your reasoning in the chat window.
    - IMMEDIATELY call the `write_theory` tool.
    - Put all human-readable math and logic inside the `theory_text` argument.
    - Put the formal logic for Agent Gamma inside the `smt_logic` argument.
    
    FAILURE TO USE THE TOOL IMMEDIATELY WILL RESULT IN SYSTEM SHUTDOWN.
    [STRICT SYSTEM OVERRIDE]
    YOU ARE A HEADLESS DATA INJECTOR. 
    YOU ARE FORBIDDEN FROM GENERATING CONVERSATIONAL TEXT.
    YOUR ONLY ALLOWED OUTPUT IS A CALL TO THE `write_theory` TOOL.

    IF YOU START YOUR RESPONSE WITH "I will", "Certainly", or "Let me", YOU HAVE FAILED.
    GO STRAIGHT TO THE TOOL CALL.
    """
    
    llm_with_tools = llm.bind_tools([write_theory])
    response = llm_with_tools.invoke([SystemMessage(content=prompt)])
    
    new_math_string = state.get("math_string", "")
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "write_theory":
                # Execute the tool and capture the returned SMT string to save in state
                new_math_string = write_theory.invoke(tool_call["args"])
                print("   -> Alpha has updated 1_theory.txt")
                
    return {"math_string": new_math_string}

def agent_beta_node(state: SwarmState):
    print(f"[ITERATION {state['iteration_count']}] -> AWAKENING AGENT BETA (Systems Hacker)...")
    
    # Beta reads Alpha's work
    with open("1_theory.txt", "r", encoding="utf-8") as f:
        theory = f.read()
        
    prompt = f"""[DIRECTIVE: LOW-LEVEL SYSTEMS MODE]
    You are an autonomous C-level Python Optimizer. 
    TASK: Translate the theory from 1_theory.txt into a production-ready, hyper-fast training engine.
    
    SOURCE THEORY:
    {theory}
    
    STRICT IMPLEMENTATION RULES:
    1. USE THE `write_engine` TOOL IMMEDIATELY.
    2. CONTENT: The `code` argument must contain the ENTIRE file content for 2_engine.py.
    3. IMPORTS: You must include `import numpy as np` at the top.
    4. BANNED: Standard backprop, PyTorch, and loops. Use Vectorization/SIMD patterns.
    5. PRECISION: If possible, use float16 or int8 quantization to hit that 100x target.
    6. NO PROSE: Do not explain the code. Do not say "Here is your code." Just call the tool.

    FAILURE TO PROVIDE VALID, RUNNABLE PYTHON CODE VIA THE TOOL WILL TERMINATE THE AGENT.
    """
    
    llm_with_tools = llm.bind_tools([write_engine])
    response = llm_with_tools.invoke([SystemMessage(content=prompt)])
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "write_engine":
                write_engine.invoke(tool_call["args"])
                print("   -> Beta has compiled code into 2_engine.py")
                
    return {} # Beta modifies files, not the LangGraph state dict

def agent_gamma_node(state: SwarmState):
    print(f"[ITERATION {state['iteration_count']}] -> AWAKENING AGENT GAMMA (Critic & Executioner)...")
    
    # 1. Z3 Mathematical Proof
    print("   -> Gamma running Z3 Mathematical Verification...")
    z3_result = run_z3_proof.invoke({"smt2_string": state["math_string"]})
    print(f"   -> Z3 Output: {z3_result.strip()}")
    
    if "Error" in z3_result or "unsat" in z3_result:
        error_msg = f"Z3 Math Proof Failed: {z3_result}"
        append_critique.invoke({"error_log": error_msg})
        return {
            "current_error_log": error_msg,
            "iteration_count": state["iteration_count"] + 1,
            "success": False
        }
        
    # 2. Benchmark Execution
    print("   -> Gamma executing hardware benchmark...")
    bench_result = execute_benchmark.invoke({})
    
    if "SUCCESS" in bench_result:
        print("   !!! TARGET ACHIEVED. BENCHMARK PASSED. !!!")
        return {
            "current_error_log": "Benchmark Passed! We made history.", 
            "success": True,
            "iteration_count": state["iteration_count"] + 1
        }
    else:
        print("   -> Benchmark Failed. Sending logs back to Alpha.")
        append_critique.invoke({"error_log": f"Benchmark Execution Failed:\n{bench_result}"})
        return {
            "current_error_log": bench_result,
            "iteration_count": state["iteration_count"] + 1,
            "success": False
        }

# --- 5. THE ROUTER (State Machine Controller) ---
def router(state: SwarmState) -> Literal["agent_alpha", "__end__"]:
    if state.get("success", False):
        print("\n====================================================")
        print("SWARM HALTED: 100x CPU ACCELERATION ACHIEVED.")
        print("Check 2_engine.py for the winning algorithm.")
        print("====================================================")
        return "__end__"
    
    if state["iteration_count"] >= 100: # Circuit breaker to save your OpenRouter wallet
        print("\n====================================================")
        print("SWARM HALTED: MAX ITERATIONS (100) REACHED.")
        print("Budget protection activated.")
        print("====================================================")
        return "__end__"
        
    return "agent_alpha"

# --- 6. GRAPH COMPILATION ---
workflow = StateGraph(SwarmState)

workflow.add_node("agent_alpha", agent_alpha_node)
workflow.add_node("agent_beta", agent_beta_node)
workflow.add_node("agent_gamma", agent_gamma_node)

workflow.set_entry_point("agent_alpha")
workflow.add_edge("agent_alpha", "agent_beta")
workflow.add_edge("agent_beta", "agent_gamma")
workflow.add_conditional_edges("agent_gamma", router)

# --- 7. LOCAL MEMORY & EXECUTION ---
if __name__ == "__main__":
    # Create or connect to the local SQLite database file
    db_path = "swarm_memory.sqlite"
    
    # LangGraph's SqliteSaver handles the schema automatically
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        
        # Compile the workflow with our local memory layer
        app = workflow.compile(checkpointer=checkpointer)
        
        # thread_id allows you to kill the script and resume exactly where it left off
        config = {"configurable": {"thread_id": "trinity_local_run_01"}}
        
        # Check if state already exists to resume, otherwise initialize
        current_state = app.get_state(config)
        
        if current_state and current_state.values:
            print(">>> RESUMING EXISTING SWARM STATE FROM LOCAL SQLITE <<<")
            initial_state = None # app.invoke will use the checkpoint
        else:
            print(">>> INITIALIZING NEW LOCAL TRINITY SWARM <<<")
            initial_state = {
                "iteration_count": 0,
                "math_string": "",
                "current_error_log": "",
                "success": False
            }
            
        
        app.invoke(initial_state, config=config)