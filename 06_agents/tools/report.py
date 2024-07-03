from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel

def write_report(filename, html):
    with open(filename, 'w') as f:
        f.write(html)

class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str

write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="write an HTML file to disk. Use this tool whenever someone asks for a report.",
    func=write_report,
)
