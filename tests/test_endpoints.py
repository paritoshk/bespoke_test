# test_endpoints.py
import asyncio
import aiohttp
import json

async def test_endpoints():
    # Test without file
    async with aiohttp.ClientSession() as session:
        print("\nTesting without file...")
        async with session.post('http://localhost:8000/train') as response:
            print(f"Response status: {response.status}")
            print(f"Response body: {await response.text()}")

        # Test with file
        print("\nTesting with file...")
        with open('examples.jsonl', 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file',
                          f,
                          filename='examples.jsonl',
                          content_type='application/json')
            
            async with session.post('http://localhost:8000/train',
                                  data=data) as response:
                print(f"Response status: {response.status}")
                print(f"Response body: {await response.text()}")

if __name__ == "__main__":
    asyncio.run(test_endpoints())