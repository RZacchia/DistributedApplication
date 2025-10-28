using BookRent.Catalog.Infrastructure.Interfaces;
using BookRent.Catalog.Model;
using Microsoft.EntityFrameworkCore;

namespace BookRent.Catalog.Infrastructure;

public class BookRepository : IBookRepository
{
    private CatalogDbContext Context { get; init; }

    public BookRepository(CatalogDbContext context)
    {
        Context = context;
    }
    
    
    public async Task<List<Book>> GetBooksAsync()
    {
        return await Context.Books.Where(b => b.IsVisible).ToListAsync();
    }

    public Task<Book?> GetBookAsync(Guid id)
    {
        return Context.Books.FirstOrDefaultAsync(b => b.Id == id);
    }

    public async Task<List<Book>> GetBooksByNameAsync(string name)
    {
        return await Context.Books.
            Where(b => b.Title.ToLower().Contains(name.ToLower()) && b.IsVisible)
            .ToListAsync();
    }

    public async Task<bool> AddBookAsync(Book book)
    {
        var alreadyExists = await Context.Books.AnyAsync(b => b.Title == book.Title);
        if (alreadyExists)
        {
            return false;
        }
        
        var addedBook = Context.Books.Add(book);
        await Context.SaveChangesAsync();
        return addedBook.Entity.Id == book.Id;
        
    }

    public async Task<bool> UpdateBookAsync(Book book)
    {
        var updated = Context.Books.Update(book);
        return await Context.SaveChangesAsync() == 1;
    }

    public async Task<bool> DeleteBookAsync(Guid id)
    {
        var book = await Context.Books.FirstOrDefaultAsync(b => b.Id == id);
        if (book == null) return false;
        Context.Books.Remove(book);
        return await Context.SaveChangesAsync() == 1;
    }
}