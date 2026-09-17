# Composio Research Agent 

An automated agentic diligence pipeline built on the Composio SDK to research software applications across categories (CRM, Support, Communications, Marketing) for API breadth, authentication methods, Model Context Protocol (MCP) support, and overall buildability.

---

## 1. Project Directory Structure

```text
C:\composio-research-agent\
│
├── .env                    # Environment variables (API keys)
├── .gitignore              # Git ignore file
├── aggregator.py           # Script to aggregate individual JSON reports into master summaries
├── all_results.json        # Consolidated dataset of all processed application research reports
├── apps_registry.py        # Target registry defining applications and categories
├── index.html              # Standalone interactive HTML case study report
├── needs_human_review.json # Log tracking applications requiring manual inspection
├── requirements.txt        # Python dependency specifications
├── research_agent.py       # Main execution script running the research loop and verification
└── results/                # Directory storing individual app JSON report outputs


Setup & Installation Instructions

"# research_agent" 
