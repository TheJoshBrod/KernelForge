import { promises as fs } from 'fs';
import path from 'path';
import { NextResponse } from 'next/server';

const PROJECTS_DIR = path.join(process.cwd(), 'projects');

export async function POST(request) {
    try {
        const formData = await request.formData();
        const projectName = formData.get('projectName');
        const code = formData.get('code');
        const weightsFile = formData.get('weightsFile'); // File object
        const validationSetPath = formData.get('validationSetPath');

        if (!projectName) {
            return NextResponse.json({ error: 'Project name is required' }, { status: 400 });
        }

        const projectPath = path.join(PROJECTS_DIR, projectName);

        // Check for duplicates
        try {
            await fs.access(projectPath);
            return NextResponse.json({ error: 'Project already exists' }, { status: 409 });
        } catch (e) {
            // Directory doesn't exist, proceed
        }

        // Create directory
        await fs.mkdir(projectPath, { recursive: true });

        // Save model.py
        if (code) {
            await fs.writeFile(path.join(projectPath, 'model.py'), code, 'utf8');
        }

        // Save weights.pt
        if (weightsFile && weightsFile.size > 0) {
            const buffer = Buffer.from(await weightsFile.arrayBuffer());
            await fs.writeFile(path.join(projectPath, 'weights.pt'), buffer);
        }

        // Save config.json
        const config = {
            validation_set: validationSetPath || null,
            created_at: new Date().toISOString()
        };
        await fs.writeFile(path.join(projectPath, 'config.json'), JSON.stringify(config, null, 2), 'utf8');

        return NextResponse.json({ success: true, name: projectName });

    } catch (error) {
        console.error('Project creation error:', error);
        return NextResponse.json({ error: 'Failed to create project' }, { status: 500 });
    }
}

export async function GET() {
    try {
        // Check if projects dir exists
        try {
            await fs.access(PROJECTS_DIR);
        } catch {
            return NextResponse.json({ projects: [] });
        }

        const entries = await fs.readdir(PROJECTS_DIR, { withFileTypes: true });

        const projects = await Promise.all(
            entries
                .filter(entry => entry.isDirectory())
                .map(async (entry) => {
                    const projectPath = path.join(PROJECTS_DIR, entry.name);
                    const stats = await fs.stat(projectPath);
                    let lastAccessed = stats.mtime.toISOString(); // Default to FS mtime

                    // Try to read config.json for last_accessed
                    try {
                        const configData = await fs.readFile(path.join(projectPath, 'config.json'), 'utf8');
                        const config = JSON.parse(configData);
                        if (config.last_accessed) {
                            lastAccessed = config.last_accessed;
                        } else if (config.created_at) {
                            // If never accessed but has created_at, could use that, but mtime is usually better fallback for "activity"
                            // However, user specifically wants "accessed". 
                            // We'll stick to: last_accessed > mtime
                        }
                    } catch (e) {
                        // Config might not exist or be readable, ignore
                    }

                    return {
                        name: entry.name,
                        lastModified: lastAccessed // We use this property for sorting on frontend/backend API contract
                    };
                })
        );

        // Sort by last modified (which is now last_accessed) descending (newest first)
        projects.sort((a, b) => new Date(b.lastModified) - new Date(a.lastModified));

        return NextResponse.json({ projects });
    } catch (error) {
        console.error('List projects error:', error);
        return NextResponse.json({ error: 'Failed to list projects' }, { status: 500 });
    }
}
