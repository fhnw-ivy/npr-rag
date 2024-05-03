### Use of AI Tools

Throughout the mini-challenge, our team used AI tools, specifically ChatGPT and GitHub Copilot, to help with the
software development aspects of our project. These tools were instrumental in handling coding and debugging tasks,
allowing us to focus on the core functionality of our RAG systems without the added overhead of
mundane coding tasks. Here we describe our approach and the strategies that were most effective.

#### **ChatGPT**.

ChatGPT was primarily used for software engineering support tasks. Due to its limitations regarding training
data cutoff, it could not directly support conceptual design decisions related to the Langchain library. However,
it proved invaluable in helping us produce well-structured, clean code and in troubleshooting programming issues.

**Prompting Strategy for ChatGPT:**

- **Specific technical questions**: We used ChatGPT for specific, technical prompts focused on generating clean and
  efficient code.
  efficient code. For example, the question "How can I create a clean class to make this evaluation logic more modular?"
  helped.
- **Debugging Assistance**: When we encountered bugs, we used detailed descriptions of the problems to solicit debugging
  advice such as "ChatGPT, why does this Python function fail when processing data frames with null values?".

#### **GitHub Copilot**

GitHub Copilot has played a critical role in speeding up the development of boilerplate and utility code, as it is
integrated directly into our directly into our IDE. This allowed us to focus on the core logic of the RAG system and the
Langchain library, and less on the repetitive and mundane aspects of software development.

**Prompting Strategy for GitHub Copilot:**

- **Inline code generation**: Most of the time, we used GitHub Copilot to generate code snippets inline. For example,
  When writing a function, we would start typing the function signature and let Copilot generate the body.
- **Code Refactoring**: Copilot was also used for code refactoring. When we came across a block of code that needed to
  be optimized or made more readable, we would ask Copilot to suggest a refactored version. This was done by simply
  writing a Python comment like `# Refactor this code for better readability`.

#### **Evaluation and Impact**.

The integration of ChatGPT and GitHub Copilot has effectively improved our software development efficiency. These tools
allowed us to automate the generation of routine code and focus our intellectual efforts on the strategic and conceptual
aspects of the RAG system. They not only saved development time, but also ensured that the code base remained clean,
maintainable, and well-documented.