import { promises as fs } from 'fs';
import path from 'path';
import { NextResponse } from 'next/server';

const configPath = path.join(process.cwd(), 'config.json');

const defaultConfig = {
    llm_info: {
        model: '',
        apikey: '',
        provider: 'anthropic', // Default provider
    }
};

export async function GET() {
    try {
        const data = await fs.readFile(configPath, 'utf8');
        return NextResponse.json(JSON.parse(data));
    } catch (error) {
        if (error.code === 'ENOENT') {
            // File doesn't exist, return default
            return NextResponse.json(defaultConfig);
        }
        return NextResponse.json({ error: 'Failed to read config' }, { status: 500 });
    }
}

export async function POST(request) {
    try {
        const body = await request.json();

        // Validate or merge if necessary, for now direct save
        await fs.writeFile(configPath, JSON.stringify(body, null, 2), 'utf8');

        return NextResponse.json({ success: true, config: body });
    } catch (error) {
        return NextResponse.json({ error: 'Failed to save config' }, { status: 500 });
    }
}
