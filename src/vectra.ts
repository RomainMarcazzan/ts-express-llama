import { LlamaEmbedding } from "node-llama-cpp";
import path from "path";
import { fileURLToPath } from "url";
import { LocalIndex } from "vectra";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const index = new LocalIndex(path.join(__dirname, "..", "index"));

export const createIndexIfNotExists = async () => {
  try {
    if (!(await index.isIndexCreated())) {
      await index.createIndex();
    }
  } catch (error) {
    const err = error as Error;
    throw new Error(`Failed to create index: ${err.message}`);
  }
};

export const resetDatabase = async () => {
  try {
    await index.deleteIndex();
    await index.createIndex({ version: 1 });
  } catch (error) {
    const err = error as Error;
    throw new Error(`Failed to reset database: ${err.message}`);
  }
};

export const visualizeVectorDB = async () => {
  try {
    const items = await index.listItems();
    console.log("Vectra Vector DB Contents:");
    items.forEach((item, idx) => {
      console.log(`Item ${idx + 1}:`);
      console.log(`ID: ${item.id}`);
      console.log(`Norm: ${item.norm}`);
      // console.log(`Vector: ${JSON.stringify(item.vector)}`);
      console.log(`Metadata: ${JSON.stringify(item.metadata)}`);
    });
  } catch (error) {
    const err = error as Error;
    throw new Error(`Failed to visualize vector DB: ${err.message}`);
  }
};

interface VectorDBItem {
  id: string;
  norm: number;
  vector?: number[];
  metadata: {
    text?: string;
    //[key: string]: any;
  };
}

export const getVectorDBContents = async (): Promise<VectorDBItem[]> => {
  try {
    const items = await index.listItems();
    return items.map(({ vector, ...rest }) => ({ ...rest, vector }));
  } catch (error) {
    const err = error as Error;
    throw new Error(`Failed to get vector DB contents: ${err.message}`);
  }
};

export const retrieveDocumentEmbeddingsFromDb = async () => {
  const items = await index.listItems();
  const embeddings = new Map<string, LlamaEmbedding>();

  await Promise.all(
    items.map(async (item) => {
      if (typeof item.metadata.text === "string") {
        const embedding = LlamaEmbedding.fromJSON({
          type: "embedding",
          vector: item.vector,
        });
        embeddings.set(item.metadata.text, embedding);
      }
    })
  );

  return embeddings;
};

export { index };
