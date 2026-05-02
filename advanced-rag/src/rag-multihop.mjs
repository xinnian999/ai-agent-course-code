import "dotenv/config";
import { z } from "zod";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { Milvus } from "@langchain/community/vectorstores/milvus";

const llm = new ChatOpenAI({
  temperature: 0,
  model: process.env.MODEL_NAME,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  apiKey: process.env.OPENAI_API_KEY,
});

const embeddings = new OpenAIEmbeddings({
  model: process.env.EMBEDDINGS_MODEL_NAME,
  dimensions: 1024,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  apiKey: process.env.OPENAI_API_KEY,
});

const ROUTE_STRATEGY = {
  simple: "simple",
  complex: "complex",
};

const NEXT_ACTION = {
  retrieve: "retrieve",
  generate: "generate",
};

const SUB_QUESTION_KEYS = ["question", "sub_question", "query", "text", "content"];

/**
 * complex：先拆解子问题序列，再按序检索
 */
const GraphState = Annotation.Root({
  question: Annotation,
  k: Annotation,
  strategy: Annotation,
  routeReason: Annotation,
  /** 拆解得到的有序子问题，仅用于检索 */
  subQuestions: Annotation,
  /** 下一轮 retrieve 要用的下标（指向 subQuestions 中尚未检索的那一条） */
  nextSubIdx: Annotation,
  documents: Annotation,
  currentQuery: Annotation,
  retrievalCount: Annotation,
  maxRetrievals: Annotation,
  plannedNext: Annotation,
  generation: Annotation,
});

let vectorStore;

async function retrieveRelevantContent(question, k) {
  try {
    const docsWithScores = await vectorStore.similaritySearchWithScore(question, k);
    return docsWithScores.map(([doc, score]) => ({
      score,
      content: doc.pageContent,
      id: doc.metadata?.id ?? "unknown",
      book_id: doc.metadata?.book_id ?? "未知",
      chapter_num: doc.metadata?.chapter_num ?? "未知",
      index: doc.metadata?.index ?? "未知",
    }));
  } catch (error) {
    console.error("检索内容时出错:", error.message);
    return [];
  }
}

/** 按 id 合并；同 id 保留更高 score */
function mergeUnique(existingDocs, newDocs) {
  const map = new Map();
  for (const d of [...existingDocs, ...newDocs]) {
    const key = String(d.id);
    const prev = map.get(key);
    if (!prev || Number(d.score) > Number(prev.score)) {
      map.set(key, d);
    }
  }
  return Array.from(map.values()).sort((a, b) => Number(b.score) - Number(a.score));
}

const RouteSchema = z.object({
  strategy: z.string(),
  reason: z.string(),
});

const NextStepSchema = z.object({
  nextAction: z.string(),
  reason: z.string(),
});

const DecomposeSchema = z.object({
  sub_questions: z.array(z.any()).min(1).max(8),
  reason: z.string(),
});

function normalizeRouteStrategy(value) {
  const raw = String(value ?? "").trim().toLowerCase();
  if (raw === ROUTE_STRATEGY.simple || raw.includes(ROUTE_STRATEGY.simple)) {
    return ROUTE_STRATEGY.simple;
  }
  if (raw === ROUTE_STRATEGY.complex || raw.includes(ROUTE_STRATEGY.complex)) {
    return ROUTE_STRATEGY.complex;
  }
  throw new Error(`route_question: 非法 strategy 输出: ${value}`);
}

function normalizeNextAction(value) {
  const raw = String(value ?? "").trim().toLowerCase();
  if (raw === NEXT_ACTION.retrieve || raw.includes(NEXT_ACTION.retrieve)) {
    return NEXT_ACTION.retrieve;
  }
  if (raw === NEXT_ACTION.generate || raw.includes(NEXT_ACTION.generate)) {
    return NEXT_ACTION.generate;
  }
  throw new Error(`plan_next_step: 非法 nextAction 输出: ${value}`);
}

function normalizeSubQuestion(item) {
  if (typeof item === "string") {
    return item.trim();
  }
  if (item && typeof item === "object") {
    for (const key of SUB_QUESTION_KEYS) {
      const value = item[key];
      if (typeof value === "string" && value.trim()) {
        return value.trim();
      }
    }
  }
  return "";
}

const routeQuestionNode = async (state) => {
  console.log("---ROUTE_QUESTION---");
  const router = llm.withStructuredOutput(RouteSchema);
  const route = await router.invoke(`
你是问答路由器。请判断用户问题是否需要外部检索，以 JSON 格式输出结果。

字段要求：
- strategy: 只能输出 simple 或 complex
- reason: 简短说明理由

规则：
- simple: 常识问答、简短定义、无需特定小说细节即可回答。
- complex: 需要《天龙八部》具体情节、人物关系、章节事实、原文细节或证据支持。

用户问题：${state.question}
`);
  const strategy = normalizeRouteStrategy(route.strategy);

  console.log(`路由策略: ${strategy} (${route.reason})`);
  return {
    strategy,
    routeReason: route.reason,
    retrievalCount: 0,
    maxRetrievals: state.maxRetrievals ?? 8,
    documents: [],
    subQuestions: [],
    nextSubIdx: 0,
    currentQuery: "",
  };
};

const decomposeQuestionNode = async (state) => {
  console.log("---DECOMPOSE_QUESTION---");
  const decomposer = llm.withStructuredOutput(DecomposeSchema);
  const out = await decomposer.invoke(`你是《天龙八部》多跳问答的「子问题拆解器」，请以 JSON 格式输出结果。

用户原始问题：
${state.question}

任务：将问题拆成**有序**子问题列表 sub_questions，用于**依次向量检索**。要求：
1. 链式推理、多层关系、因果先后的问题，必须拆成多条；单跳即可答的也可只输出 1 条。
2. 每条子问题必须是**可独立检索**的完整中文问句，**禁止**使用「他/她/此人/上文」等指代；可写全人物名与事件名。
3. 顺序必须符合推理链：先搞清前置实体/事实，再查后续结论。
4. **不要**把整句原题原样复制成唯一一条（除非确实无法拆分）；不要拆成过碎的关键词列表。
5. 输出 1～8 条即可。
6. sub_questions 数组中的每一项都必须是字符串，不要输出对象。

请输出 sub_questions 与简短 reason。`);

  const subQuestions = out.sub_questions.map(normalizeSubQuestion).filter(Boolean);
  if (subQuestions.length === 0) {
    throw new Error("decompose_question: sub_questions 为空");
  }

  console.log(`拆解 ${subQuestions.length} 条子问题 (${out.reason})`);
  subQuestions.forEach((q, i) => {
    console.log(`  [${i + 1}] ${q}`);
  });

  return {
    subQuestions,
    nextSubIdx: 0,
    currentQuery: subQuestions[0],
  };
};

const retrieveNode = async (state) => {
  const subs = state.subQuestions ?? [];
  const idx = state.nextSubIdx ?? 0;
  const q = subs[idx]?.trim();
  if (!q) {
    throw new Error(`retrieve: 子问题下标 ${idx} 无有效文本（共 ${subs.length} 条）`);
  }

  const round = state.retrievalCount + 1;
  console.log(`---RETRIEVE (第 ${round} 轮，子问题 ${idx + 1}/${subs.length})---`);
  console.log(`查询: ${q}`);

  const newDocs = await retrieveRelevantContent(q, state.k);
  const merged = mergeUnique(state.documents ?? [], newDocs);

  if (newDocs.length === 0) {
    console.log("本轮未命中文档");
  } else {
    console.log(`本轮命中 ${newDocs.length} 条，累计去重后 ${merged.length} 条`);
    newDocs.forEach((item, i) => {
      const preview =
        item.content.length > 120 ? `${item.content.substring(0, 120)}...` : item.content;
      console.log(
        `[R${i + 1}] score=${Number(item.score).toFixed(4)} chapter=${item.chapter_num} index=${item.index}`,
      );
      console.log(`      ${preview}`);
    });
  }

  return {
    documents: merged,
    retrievalCount: round,
    nextSubIdx: idx + 1,
    currentQuery: q,
  };
};

const planNextStepNode = async (state) => {
  console.log("---PLAN_NEXT_STEP---");
  const subs = state.subQuestions ?? [];
  const nextIdx = state.nextSubIdx ?? 0;
  const remaining = subs.length - nextIdx;

  const subList = subs.map((s, i) => `${i + 1}. ${s}${i < nextIdx ? " （已检索）" : i === nextIdx ? " （下一轮将检索，若选择继续）" : " （未检索）"}`).join("\n");

  const docStr =
    state.documents.length === 0
      ? "（尚无检索结果）"
      : state.documents
        .slice(0, 6)
        .map(
          (d, i) =>
            `[${i + 1}] score=${Number(d.score).toFixed(4)} 第${d.chapter_num}章: ${d.content.slice(0, 200)}${d.content.length > 200 ? "..." : ""}`,
        )
        .join("\n\n");

  const prompt = `你是多跳 RAG 规划器，请以 JSON 格式输出结果。检索查询已由前置步骤拆解为**有序子问题**；若需继续检索，下一轮将自动使用「下一条子问题」做向量检索，你**不要**自拟新的检索句。

字段要求：
- nextAction: 只能输出 retrieve 或 generate
- reason: 简短说明理由

用户原始问题：${state.question}

子问题序列：
${subList || "（无）"}

已检索轮数：${state.retrievalCount}；剩余未检索子问题条数：${remaining}
最大检索轮数上限：${state.maxRetrievals}

已召回文档摘要：
${docStr}

请判断下一步：
1) 已有足够依据回答用户原始问题 → nextAction=generate
2) 仍缺关键事实、且仍存在未检索的子问题、且未超过轮数上限 → nextAction=retrieve

硬性规则：
- 若剩余未检索子问题条数为 0，必须 nextAction=generate。
- 若已检索轮数已达到或超过最大检索轮数，必须 nextAction=generate。`;

  const model = llm.withStructuredOutput(NextStepSchema);
  const out = await model.invoke(prompt);
  const nextAction = normalizeNextAction(out.nextAction);

  let finalNext = nextAction;
  if (state.retrievalCount >= state.maxRetrievals) finalNext = NEXT_ACTION.generate;
  if (remaining <= 0) finalNext = NEXT_ACTION.generate;

  console.log(`[决策] plannedNext=${finalNext} (模型建议=${nextAction}) (${out.reason})`);

  return {
    plannedNext: finalNext,
  };
};

function afterRoute(state) {
  return state.strategy === ROUTE_STRATEGY.simple ? "direct_answer" : "decompose_question";
}

function afterPlan(state) {
  return state.plannedNext === NEXT_ACTION.retrieve ? "retrieve" : "generate";
}

const directAnswerNode = async (state) => {
  console.log("---DIRECT_ANSWER---");
  process.stdout.write("\n【AI 回答（流式）】\n");
  let generation = "";
  const stream = await llm.stream(`你是一个中文问答助手，请直接简洁回答问题。

问题：${state.question}
`);
  for await (const chunk of stream) {
    const text = typeof chunk.content === "string" ? chunk.content : "";
    if (!text) continue;
    generation += text;
    process.stdout.write(text);
  }
  process.stdout.write("\n");
  return { generation };
};

const generateNode = async (state) => {
  console.log("---GENERATE---");
  const context = state.documents
    .map(
      (item, i) =>
        `[片段 ${i + 1}]
章节: 第 ${item.chapter_num} 章
内容: ${item.content}`,
    )
    .join("\n\n━━━━━\n\n");
  process.stdout.write("\n【AI 回答（流式）】\n");
  let generation = "";

  const stream = await llm.stream(`你是一个中文小说问答助手。请仅依据给定检索片段，回答用户问题。

要求：
1. 若片段足以回答，则先给出简洁结论，再用片段中的关键信息支撑。
2. 若片段不足以完全确定，请明确说明“不确定/片段不足以确认”，并给出最接近的依据。
3. 不要编造未出现在片段中的情节。

用户问题：${state.question}

检索片段：
${context || "（无检索结果）"}
`);

  for await (const chunk of stream) {
    const text = typeof chunk.content === "string" ? chunk.content : "";
    if (!text) continue;
    generation += text;
    process.stdout.write(text);
  }
  process.stdout.write("\n");

  return { generation };
};

const graph = new StateGraph(GraphState)
  .addNode("route_question", routeQuestionNode)
  .addNode("direct_answer", directAnswerNode)
  .addNode("decompose_question", decomposeQuestionNode)
  .addNode("retrieve", retrieveNode)
  .addNode("plan_next_step", planNextStepNode)
  .addNode("generate", generateNode)
  .addEdge(START, "route_question")
  .addConditionalEdges("route_question", afterRoute, {
    direct_answer: "direct_answer",
    decompose_question: "decompose_question",
  })
  .addEdge("decompose_question", "retrieve")
  .addEdge("retrieve", "plan_next_step")
  .addConditionalEdges("plan_next_step", afterPlan, {
    retrieve: "retrieve",
    generate: "generate",
  })
  .addEdge("direct_answer", END)
  .addEdge("generate", END)
  .compile();

async function initVectorStore() {
  console.log("连接到 Milvus...");
  vectorStore = await Milvus.fromExistingCollection(embeddings, {
    collectionName: "ebook_collection",
    url: "localhost:19530",
    textField: "content",
    primaryField: "id",
    vectorField: "vector",
    indexCreateOptions: {
      metric_type: "COSINE",
      index_type: "HNSW",
      params: { M: 16, efConstruction: 200 },
      search_params: { ef: 64 },
    },
  });
  vectorStore.indexSearchParams = { metric_type: "COSINE", params: JSON.stringify({ ef: 64 }) };
  console.log("✓ 已连接");

  try {
    await vectorStore.client.loadCollection({ collection_name: "ebook_collection" });
    console.log("\n✓ 集合 ebook_collection 已加载");
  } catch (error) {
    if (!error.message.includes("already loaded")) {
      throw error;
    }
    console.log("\n✓ 集合 ebook_collection 已处于加载状态");
  }
}

async function main() {
  await initVectorStore();

  const question = "《天龙八部》中「四大恶人」排行第二的是谁？此人之子在身世揭晓前，其生父在武林中的公开身份是什么？";
  console.log("\n" + "=".repeat(80));
  console.log(`问题: ${question}`);
  console.log("=".repeat(80));

  const result = await graph.invoke({
    question,
    k: 3,
    strategy: "",
    routeReason: "",
    documents: [],
    subQuestions: [],
    nextSubIdx: 0,
    currentQuery: "",
    retrievalCount: 0,
    maxRetrievals: 8,
    plannedNext: "",
    generation: "",
  });

  if (result.strategy === "complex") {
    if (result.subQuestions?.length) {
      console.log("\n【子问题序列】");
      result.subQuestions.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
    }
    console.log("\n【检索相关内容（累计）】");
    if (result.documents.length === 0) {
      console.log("未找到相关内容");
    } else {
      result.documents.forEach((item, i) => {
        console.log(`\n[片段 ${i + 1}] 相似度: ${Number(item.score).toFixed(4)}`);
        console.log(`书籍: ${item.book_id}`);
        console.log(`章节: 第 ${item.chapter_num} 章`);
        console.log(`片段索引: ${item.index}`);
        console.log(
          `内容: ${item.content.substring(0, 200)}${item.content.length > 200 ? "..." : ""}`,
        );
      });
    }
    console.log(`\n检索轮数: ${result.retrievalCount} / ${result.maxRetrievals}`);
  }

  console.log(`\n最终策略: ${result.strategy}`);
  if (!result.generation?.trim()) {
    console.log("模型未返回内容。");
  }
}

main().catch((err) => {
  console.error("运行失败:", err);
  process.exit(1);
});
