import { kv } from '@vercel/kv'
import { Pinecone, RecordMetadata, ScoredPineconeRecord } from '@pinecone-database/pinecone';
import { OpenAIStream, StreamingTextResponse } from 'ai'
import OpenAI from 'openai'

import { auth } from '@/auth'
import { nanoid } from '@/lib/utils'

export const runtime = 'edge'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
})

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!
})
const index = pc.Index("questions")

const searchVectorDb = async (question: string) => {
  console.log("creating embedding for question", question);
  const embeddingRes = await openai.embeddings.create({input: question, model: "text-embedding-3-small"});
  const embedding = embeddingRes.data[0].embedding;
  console.log("Querrying Pinecone DB");
  const queryResponse = await index.query({
    vector: embedding,
    topK: 3,
    includeMetadata: true,
    includeValues: false
  })
  console.log("Pinecone response: ", queryResponse)
  return queryResponse.matches;
}

const generateSystemPrompt = (similarQuestions: ScoredPineconeRecord<RecordMetadata>[]) => {
  let content = "Based on the user's question, here are some similar results from YCombinator's Youtube Videos. Please reference these directly, including the Youtube Video Name, when answering the user's questions."
  similarQuestions.forEach((value) => {
    const metadata = value.metadata;
    if (!metadata) return;
    content += `\n\nYoutube Video Name: ${metadata.video_name} \nYoutube Video URL: ${metadata.youtube_url}\n Question: ${metadata.question} \nAnswer: ${metadata.answer}`
  })
  const systemPrompt = {
    role: "system",
    content: content
  }
  return systemPrompt;
}

export async function POST(req: Request) {
  const json = await req.json()
  let { messages, previewToken } = json
  const userId = (await auth())?.user.id

  if (!userId) {
    return new Response('Unauthorized', {
      status: 401
    })
  }

  if (previewToken) {
    openai.apiKey = previewToken
  }

  if (messages.length > 1) {
    const most_recent_message = messages[messages.length - 1];
    const similarQuestions = await searchVectorDb(most_recent_message);
    const systemPrompt = generateSystemPrompt(similarQuestions);
    messages = [...messages, systemPrompt];
  }


  const res = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages,
    temperature: 0.3,
    stream: true
  })

  const stream = OpenAIStream(res, {
    async onCompletion(completion) {
      const title = json.messages[0].content.substring(0, 100)
      const id = json.id ?? nanoid()
      const createdAt = Date.now()
      const path = `/chat/${id}`
      const payload = {
        id,
        title,
        userId,
        createdAt,
        path,
        messages: [
          ...messages,
          {
            content: completion,
            role: 'assistant'
          }
        ]
      }
      await kv.hmset(`chat:${id}`, payload)
      await kv.zadd(`user:chat:${userId}`, {
        score: createdAt,
        member: `chat:${id}`
      })
    }
  })

  return new StreamingTextResponse(stream)
}

const SYSTEM_PROMPT = "The person asking questions is ambitious and looking to start a company Use the provided context if it makes sense. If not ask the user to ask again with more detail. Here is some context based on ycombinator's youtube videos. Context: ";