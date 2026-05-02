import "dotenv/config";
import { MilvusClient, MetricType } from '@zilliz/milvus2-sdk-node';
import { OpenAIEmbeddings } from "@langchain/openai";

const COLLECTION_NAME = 'ai_diary';
const VECTOR_DIM = 1024;

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.EMBEDDINGS_MODEL_NAME,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL
  },
  dimensions: VECTOR_DIM
});

const client = new MilvusClient({
  address: 'localhost:19530'
});

async function getEmbedding(text) {
  const result = await embeddings.embedQuery(text);
  return result;
}

async function main() {
  try {
    console.log('Connecting to Milvus...');
    await client.connectPromise;
    console.log('✓ Connected\n');

    // 向量搜索
    console.log('Searching for similar diary entries...');
    const query = '我做饭或学习的日记';
    console.log(`Query: "${query}"\n`);

    const subQueries = query.includes('或')
      ? query.split('或').map(s => s.trim()).filter(Boolean)
      : [query];

    const seen = new Map();
    for (const q of subQueries) {
      const vec = await getEmbedding(q);
      const result = await client.search({
        collection_name: COLLECTION_NAME,
        vector: vec,
        limit: 2,
        metric_type: MetricType.COSINE,
        output_fields: ['id', 'content', 'date', 'mood', 'tags']
      });
      for (const item of result.results) {
        if (!seen.has(item.id) || seen.get(item.id).score < item.score) {
          seen.set(item.id, item);
        }
      }
    }

    const results = [...seen.values()].sort((a, b) => b.score - a.score).slice(0, 2);
    console.log(`Found ${results.length} results:\n`);
    results.forEach((item, index) => {
      console.log(`${index + 1}. [Score: ${item.score.toFixed(4)}]`);
      console.log(`   ID: ${item.id}`);
      console.log(`   Date: ${item.date}`);
      console.log(`   Mood: ${item.mood}`);
      console.log(`   Tags: ${item.tags?.join(', ')}`);
      console.log(`   Content: ${item.content}\n`);
    });

  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
