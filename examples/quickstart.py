"""Quickstart: a thinking, feeling, learning mind in 15 lines."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognify import MindBuilder

# 1. Assemble a mind with a personality preset
mind = (MindBuilder("Aria")
        .with_preset("curious_child")
        .enable_tool_use()
        .build())

# 2. Teach it facts in plain English
mind.teach("Birds are animals. Birds can fly. Birds have wings.")
mind.teach("Animals are living things.")

# 3. Ask questions - it answers with inference chains
print(mind.ask("What is a bird?"))
print(mind.ask("Can birds fly?"))
print(mind.ask("Why can birds fly?"))

# 4. It feels - praise raises joy, threats raise fear
reply = mind.process("I love this, you are wonderful!")
print(reply.meta["dominant_emotion"], "->", reply.text)

# 5. It uses tools when arithmetic appears
print(mind.ask("What is 12 * 12?"))

# 6. It learns - the Q-agent records which reply strategies pay off
mind.feedback(+0.8)                     # reward the last exchange
print(mind.stats()["learning"]["q_learning"])
