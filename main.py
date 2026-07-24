import asyncio
from dotenv import load_dotenv
load_dotenv()  
from memory_engine.client import MnemosyneClient, MnemosyneConfig

async def main():
    client = await MnemosyneClient.create(MnemosyneConfig(
        mongo_url="mongodb://agent:agent@localhost:27018/memories?authSource=admin",
        qdrant_url="http://localhost:6334",
    ))
    
    
    result = await client.turn(user_id="persona_01", query="hello.")

    print(result)

if __name__ == "__main__":
    asyncio.run(main())