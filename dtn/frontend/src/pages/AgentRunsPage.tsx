import { AgentWorkflowPanel } from "./AssistantPage";

interface Props {
  token: string;
  active: boolean;
}

export default function AgentRunsPage({ token, active }: Props) {
  return (
    <section className="content agent-runs-page">
      <div className="page-heading">
        <h1>Agent runs</h1>
      </div>
      {active && <AgentWorkflowPanel token={token} />}
    </section>
  );
}
