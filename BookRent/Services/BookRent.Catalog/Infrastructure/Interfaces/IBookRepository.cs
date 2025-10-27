using BookRent.Catalog.Model;

namespace BookRent.Catalog.Infrastructure.Interfaces;

public interface IBookRepository
{
    public Task<List<Book>> GetBooksAsync();
    public Task<Book?> GetBookAsync(Guid id);
    public Task<List<Book>> GetBooksByNameAsync(string name);
    public Task<bool> AddBookAsync(Book book);
    public Task<bool> UpdateBookAsync(Book book);
    public Task<bool> DeleteBookAsync(Book book);
}