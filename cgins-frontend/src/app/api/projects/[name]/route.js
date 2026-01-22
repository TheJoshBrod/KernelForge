import { promises as fs } from 'fs';
import path from 'path';
import { NextResponse } from 'next/server';

const PROJECTS_DIR = path.join(process.cwd(), 'projects');

export async function DELETE(request, { params }) {
    const { name } = await params;

    if (!name) {
        return NextResponse.json({ error: 'Project name is required' }, { status: 400 });
    }

    const projectPath = path.join(PROJECTS_DIR, name);

    try {
        // Check if exists
        await fs.access(projectPath);

        // Delete directory recursively
        await fs.rm(projectPath, { recursive: true, force: true });

        return NextResponse.json({ success: true, message: `Project ${name} deleted` });
    } catch (error) {
        if (error.code === 'ENOENT') {
            return NextResponse.json({ error: 'Project not found' }, { status: 404 });
        }
        console.error('Delete project error:', error);
        return NextResponse.json({ error: 'Failed to delete project' }, { status: 500 });
    }
}

export async function PATCH(request, { params }) {
    const { name } = await params;
    if (!name) {
        return NextResponse.json({ error: 'Project name is required' }, { status: 400 });
    }

    const projectPath = path.join(PROJECTS_DIR, name);
    const configPath = path.join(projectPath, 'config.json');

    try {
        // Read existing config or create new object if missing
        let config = {};
        try {
            const data = await fs.readFile(configPath, 'utf8');
            config = JSON.parse(data);
        } catch (e) {
            // Config missing, start fresh
        }

        // Update timestamp
        config.last_accessed = new Date().toISOString();

        // Write back
        await fs.writeFile(configPath, JSON.stringify(config, null, 2), 'utf8');

        return NextResponse.json({ success: true, last_accessed: config.last_accessed });
    } catch (error) {
        console.error('Update project accessed time error:', error);
        return NextResponse.json({ error: 'Failed to update project' }, { status: 500 });
    }
}
