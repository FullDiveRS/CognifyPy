"""Save a whole mind - facts, memories, emotions, neural weights - to SQLite."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognify import MindBuilder
from cognify.persistence.store import SqliteMindStore

mind = (MindBuilder("Persisto").with_seed(1)
        .enable_tool_use().build())
mind.teach("Cats are animals. Cats can purr.")
mind.process("I love cats, they are wonderful!")
mind.process("What is 9 * 9?")
mind.feedback(+1.0)

store = SqliteMindStore("minds.db")     # a single SQLite file
store.save(mind, "persisto-v1")
print("saved:", [s["name"] for s in store.list()])

# ...later, in another process or after a restart...
clone = store.load("persisto-v1")
print("clone remembers:", clone.ask("Can cats purr?"))
print("clone knows", clone.kb.size(), "facts and has lived",
      clone.cycle_count, "cycles")
print("neural weights survived:",
      clone.learner.net.state_dict()["w1"] == mind.learner.net.state_dict()["w1"])

store.delete("persisto-v1")
store.close()
